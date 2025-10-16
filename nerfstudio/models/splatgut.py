"""
SplatGUT: 相机用独立 SH + 3DGUT，LiDAR 用 features + decoder

Good Taste（Linus 认证 + Debug 版本）:
    1. 单一渲染路径：Camera → rays → 3DGUT
    2. RayBundle 只是"查找 Camera"的线索
    3. 详细 debug 信息追踪问题
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.cameras.rays import RayBundle

@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    """SplatGUT 配置."""
    _target: Type = field(default_factory=lambda: SplatGUTModel)

    use_ray_tracing: bool = True
    sh_degree: int = 3
    target_radius: float = 200.0
    k_buffer_size: int = 32
    ut_alpha: float = 1.0
    ut_beta: float = 0.0
    camera_model: str = "pinhole"
    fisheye_distortion: Optional[list] = None
    use_camopt_in_eval: bool = True


class SplatGUTModel(SplatADModel):
    """SplatGUT: 相机用 SH + 3DGUT."""
    
    config: SplatGUTModelConfig

    def populate_modules(self):
        """初始化."""
        super().populate_modules()
        
        self._init_independent_sh()  # ← 改名
        self._disable_camera_decoder()
        
        if self.config.use_ray_tracing:
            self._init_gut_renderer()
        
        # 惰性初始化
        self.train_cameras = None
        self._train_cameras_extracted = False

    def _init_independent_sh(self):
        """初始化独立 SH - 第一次."""
        means = self.gauss_params['means']  # ← 直接访问，不用循环
        
        N, device, dtype = means.shape[0], means.device, means.dtype
        L = self.config.sh_degree
        M = (L + 1) ** 2
        
        self.sh_coeffs_dc = nn.Parameter(
            torch.zeros(N, 3, 1, device=device, dtype=dtype)
        )
        self.sh_coeffs_rest = nn.Parameter(
            torch.zeros(N, 3, M - 1, device=device, dtype=dtype)
        )
        
        self._last_known_N = N  # ← 新增：记录当前点数
        print(f"[SplatGUT] Independent SH initialized: N={N}, L={L}")

    def _disable_camera_decoder(self):
        """禁用相机 decoder."""
        if hasattr(self, 'rgb_decoder'):
            self.rgb_decoder = nn.Identity()
            print("[SplatGUT] Disabled rgb_decoder")
        
        # if hasattr(self, 'appearance_embedding'):
        #     self.appearance_embedding = nn.Identity()

    def _sync_sh_if_needed(self):
        """
        自动同步 SH：完全断开旧 graph，创建新 Parameter.
        
        Linus: "Clean break, no half-measures!"
        """
        N_cur = self.gauss_params['means'].shape[0]
        N_sh = self.sh_coeffs_dc.shape[0]
        
        if N_cur == N_sh:
            return
        
        print(f"[SplatGUT] Densify detected! {N_sh} → {N_cur}, syncing SH...")
        
        device = self.sh_coeffs_dc.device
        dtype = self.sh_coeffs_dc.dtype
        M = self.sh_coeffs_rest.shape[-1] + 1
        
        # ============================================================
        # 关键修复：完全断开旧连接，创建新 Parameter
        # ============================================================
        with torch.no_grad():
            if N_cur > N_sh:
                # 扩展
                new_dc = torch.zeros(N_cur, 3, 1, device=device, dtype=dtype)
                new_rest = torch.zeros(N_cur, 3, M - 1, device=device, dtype=dtype)
                
                # 复制旧数据（detach 确保断开连接）
                new_dc[:N_sh] = self.sh_coeffs_dc.detach()
                new_rest[:N_sh] = self.sh_coeffs_rest.detach()
                
            else:
                # 裁剪
                print(f"[SplatGUT] WARNING: Pruning to {N_cur}")
                new_dc = self.sh_coeffs_dc[:N_cur].detach().clone()
                new_rest = self.sh_coeffs_rest[:N_cur].detach().clone()
        
        # 先删除旧的（释放内存和 autograd 引用）
        del self.sh_coeffs_dc
        del self.sh_coeffs_rest
        
        # 清理 CUDA 缓存（可选但推荐）
        torch.cuda.empty_cache()
        
        # 创建新 Parameter（全新的 autograd 对象）
        self.sh_coeffs_dc = nn.Parameter(new_dc.contiguous(), requires_grad=True)
        self.sh_coeffs_rest = nn.Parameter(new_rest.contiguous(), requires_grad=True)
        
        self._update_optimizer_for_sh()
        
        self._last_known_N = N_cur
        print(f"[SplatGUT] SH synced and re-registered to optimizer")

    def _update_optimizer_for_sh(self):
        """
        更新优化器：移除旧的 SH 参数，添加新的.
        
        Linus: "When you change the Parameter, you MUST update the optimizer!"
        """
        # 获取优化器（从 trainer 传入）
        if not hasattr(self, 'optimizers') or self.optimizers is None:
            print("[SplatGUT] WARNING: No optimizers found, SH may not be trained!")
            return
        
        # SplatGUT 的 SH 在 'camera_sh' 组（根据 get_param_groups）
        if 'camera_sh' not in self.optimizers.optimizers:
            print("[SplatGUT] WARNING: 'camera_sh' optimizer not found!")
            return
        
        optimizer = self.optimizers.optimizers['camera_sh']
        
        # 移除旧的参数组
        # 注意：需要清除 state（momentum buffer 等）
        optimizer.param_groups = []
        optimizer.state.clear()
        
        # 添加新参数组
        lr = self.optimizers.config['camera_sh']['optimizer'].lr
        optimizer.add_param_group({
            'params': [self.sh_coeffs_dc, self.sh_coeffs_rest],
            'lr': lr,
            'name': 'camera_sh',
        })
        
        print(f"[SplatGUT] Optimizer updated: {len(optimizer.param_groups)} groups")

    def _init_gut_renderer(self):
        """初始化 3DGUT."""
        try:
            from nerfstudio.adapters.threedgut_adapter import GUT3DRenderer, THREEDGUT_AVAILABLE
        except ImportError as e:
            print(f"[SplatGUT] 3DGUT import failed: {e}")
            self.config.use_ray_tracing = False
            return
        
        if not THREEDGUT_AVAILABLE:
            self.config.use_ray_tracing = False
            return
        
        gut_config = {
            'render': {
                'method': '3dgut',
                'particle_radiance_sph_degree': self.config.sh_degree,
                'particle_kernel_degree': 2,
                'particle_kernel_min_response': 0.0,
                'particle_kernel_min_alpha': 0.0,
                'particle_kernel_max_alpha': 1.0,
                'min_transmittance': 0.001,
                'enable_hitcounts': True,
                'splat': {
                    'n_rolling_shutter_iterations': 1,
                    'k_buffer_size': self.config.k_buffer_size,
                    'global_z_order': False,
                    'ut_alpha': self.config.ut_alpha,
                    'ut_beta': self.config.ut_beta,
                    'ut_kappa': 0.0,
                    'ut_in_image_margin_factor': 1.5,
                    'ut_require_all_sigma_points_valid': False,
                    'rect_bounding': True,
                    'tight_opacity_bounding': True,
                    'tile_based_culling': True,
                }
            }
        }
        
        self.gut_renderer = GUT3DRenderer(gut_config)
        print(f"[SplatGUT] 3DGUT ready")

    def get_outputs(self, sensor: Union[RayBundle, Cameras, Lidars]) -> Dict[str, Tensor]:
        """
        主入口 + DEBUG.
        """
        # # DEBUG: 详细类型信息
        # print(f"\n[DEBUG] ========== get_outputs called ==========")
        # print(f"[DEBUG] sensor type: {type(sensor)}")
        # print(f"[DEBUG] sensor.__class__: {sensor.__class__}")
        # print(f"[DEBUG] sensor.__class__.__name__: {sensor.__class__.__name__}")
        # print(f"[DEBUG] sensor.__class__.__module__: {sensor.__class__.__module__}")
        # print(f"[DEBUG] isinstance(sensor, RayBundle): {isinstance(sensor, RayBundle)}")
        # print(f"[DEBUG] isinstance(sensor, Cameras): {isinstance(sensor, Cameras)}")
        # print(f"[DEBUG] isinstance(sensor, Lidars): {isinstance(sensor, Lidars)}")
        # print(f"[DEBUG] RayBundle class: {RayBundle}")
        # print(f"[DEBUG] Cameras class: {Cameras}")
        if hasattr(self, 'sh_coeffs_dc'):  # 只在相机渲染时需要
            self._sync_sh_if_needed()
        # LiDAR → 父类
        if isinstance(sensor, Lidars):
            print("[DEBUG] → Lidars branch")
            return super().get_outputs(sensor)
        
        # 相机路径
        if self.config.use_ray_tracing:
            if isinstance(sensor, RayBundle):
                # 训练：提取 camera_idx
                # print("[DEBUG] → RayBundle branch (TRAINING)")
                cam_idx = self._extract_camera_idx_from_raybundle(sensor)
                # print(f"[DEBUG] Extracted cam_idx: {cam_idx}")
                return self._render_camera_with_gut(cam_idx)
            
            elif isinstance(sensor, Cameras):
                # 评估：直接用 Cameras
                # print("[DEBUG] → Cameras branch (EVAL)")
                return self._render_camera_with_gut(sensor)
        
        # 回退
        # print("[DEBUG] → Fallback to parent")
        return super().get_outputs(sensor)

    def _extract_camera_idx_from_raybundle(self, ray_bundle: RayBundle) -> int:
        """从 RayBundle 提取 camera_idx."""
        if ray_bundle.camera_indices is None:
            raise ValueError("RayBundle must have camera_indices")
        
        cam_idx = int(ray_bundle.camera_indices.flatten()[0].item())
        # print(f"[DEBUG] _extract_camera_idx_from_raybundle: cam_idx={cam_idx}")
        
        # 惰性初始化训练相机
        if not self._train_cameras_extracted:
            self._extract_train_cameras()
        
        # 检查索引有效性
        if self.train_cameras is None or cam_idx >= len(self.train_cameras):
            raise RuntimeError(
                f"Camera {cam_idx} not found. "
                f"train_cameras has {len(self.train_cameras) if self.train_cameras else 0} cameras."
            )
        
        return cam_idx

    def _extract_train_cameras(self):
        """惰性提取训练相机."""
        print("[DEBUG] _extract_train_cameras called")
        print(f"[DEBUG] self.kwargs keys: {list(self.kwargs.keys())}")
        
        if 'train_dataset' in self.kwargs:
            dataset = self.kwargs['train_dataset']
            print(f"[DEBUG] Found train_dataset: {type(dataset)}")
            print(f"[DEBUG] train_dataset attributes: {dir(dataset)}")
            
            if hasattr(dataset, 'cameras'):
                self.train_cameras = dataset.cameras
                # print(f"[DEBUG] train_cameras type: {type(self.train_cameras)}")
                # print(f"[DEBUG] train_cameras length: {len(self.train_cameras)}")
                # print(f"[DEBUG] train_cameras class: {self.train_cameras.__class__}")
                self._train_cameras_extracted = True
                return
        
        raise RuntimeError(
            "[SplatGUT] Cannot find training cameras!\n"
            "\n"
            "Solution: Modify Pipeline:\n"
            "    self._model = config.model.setup(..., train_dataset=self.datamanager.train_dataset)\n"
        )

    def _render_camera_with_gut(self, camera_or_idx: Union[Cameras, int]) -> Dict[str, Tensor]:
        """统一渲染路径 + DEBUG."""
        # print(f"\n[DEBUG] ========== _render_camera_with_gut called ==========")
        # print(f"[DEBUG] camera_or_idx type: {type(camera_or_idx)}")
        # print(f"[DEBUG] isinstance(camera_or_idx, int): {isinstance(camera_or_idx, int)}")
        
        # Step 1: 获取 Camera 对象
        if isinstance(camera_or_idx, int):
            # 训练：从索引获取 Camera
            print(f"[DEBUG] Getting camera from index {camera_or_idx}")
            print(f"[DEBUG] self.train_cameras type: {type(self.train_cameras)}")
            
            # 尝试不同的索引方式
            print(f"[DEBUG] Trying self.train_cameras[{camera_or_idx}]...")
            camera_attempt1 = self.train_cameras[camera_or_idx]
            print(f"[DEBUG] Result type: {type(camera_attempt1)}")
            
            print(f"[DEBUG] Trying self.train_cameras[{camera_or_idx}:{camera_or_idx+1}]...")
            camera_attempt2 = self.train_cameras[camera_or_idx:camera_or_idx+1]
            print(f"[DEBUG] Result type: {type(camera_attempt2)}")
            
            # 使用切片方式（正确的方式！）
            camera = self.train_cameras[camera_or_idx:camera_or_idx+1]
            camera = camera.to(self.device)
            print(f"[DEBUG] Final camera type: {type(camera)}")
        else:
            # 评估：直接用传入的 Cameras
            print(f"[DEBUG] Using passed Cameras object")
            camera = camera_or_idx
        
        # print(f"[DEBUG] camera type before optimization: {type(camera)}")
        # print(f"[DEBUG] camera.__class__: {camera.__class__}")
        
        # # Step 2: 应用相机优化
        # if self.training or self.config.use_camopt_in_eval:
        #     camera = camera.camera_to_worlds
        #     print(f"[DEBUG] camera type after optimization: {type(camera)}")
        
        # Step 3: 降采样
        scale = self._get_downscale_factor()
        # print(f"[DEBUG] downscale_factor: {scale}")
        
        if scale != 1:
            # print(f"[DEBUG] Rescaling with factor {1/scale}")
            # print(f"[DEBUG] camera has rescale_output_resolution? {hasattr(camera, 'rescale_output_resolution')}")
            camera.rescale_output_resolution(1 / scale)
        
        W, H = int(camera.width.item()), int(camera.height.item())
        c2w = camera.camera_to_worlds
        
        # print(f"[DEBUG] W={W}, H={H}")
        # print(f"[DEBUG] c2w shape: {c2w.shape}")
        
        # Step 4: 生成射线
        rays_o, rays_d = self._generate_rays(camera, W, H, c2w)
        
        # Step 5: 构建 gauss_params（用 SH 替换 features）
        gauss_params_sh = self._build_gauss_params_with_sh()
        
        # Step 6: 3DGUT 渲染
        distortion = None
        if hasattr(camera, 'distortion_params') and camera.distortion_params is not None:
            distortion = camera.distortion_params.cpu().numpy()
        
        outputs = self.gut_renderer.render(
            gauss_params=gauss_params_sh,
            camera=camera,
            rays_o=rays_o,
            rays_d=rays_d,
            c2w=c2w,
            target_radius=self.config.target_radius,
            gut_sh_degree=self.config.sh_degree,
            camera_model=self.config.camera_model,
            distortion=distortion,
            training=self.training,
        )

        # ====== 补齐 SplatAD 需要的 self.info（新增） ======
        depth_t = outputs["depth"]
        alpha_t = outputs["alpha"]
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(-1)
        # 语义对齐的近似：已“实心”(alpha>0.5) 像素用其深度作为中位深度，其余置0
        median_depths = torch.where(alpha_t > 0.5, depth_t, torch.zeros_like(depth_t))

        self.info = {
            "height": torch.tensor(H, device=depth_t.device),
            "width":  torch.tensor(W, device=depth_t.device),
            "median_depths": median_depths,  # [H, W, 1]
        }
        # ===============================================

        # Step 7: 还原分辨率
        if scale != 1:
            camera.rescale_output_resolution(scale)
        
        # Step 8: 后处理
        return self._postprocess_outputs(outputs, H, W)

    def _generate_rays(self, camera: Cameras, W: int, H: int, c2w: Tensor) -> tuple:
        """生成射线."""
        device = c2w.device
        fx, fy = float(camera.fx.item()), float(camera.fy.item())
        cx, cy = float(camera.cx.item()), float(camera.cy.item())
        
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        R = c2w[0, :3, :3] if c2w.dim() == 3 else c2w[:3, :3]
        t = c2w[0, :3, 3] if c2w.dim() == 3 else c2w[:3, 3]
        
        is_fisheye = (
            self.config.camera_model == "fisheye" or
            (self.config.camera_model == "auto" and 
             hasattr(camera, 'distortion_params') and 
             camera.distortion_params is not None)
        )
        
        if is_fisheye:
            dirs_cam = self._fisheye_unproject(xx, yy, fx, fy, cx, cy, camera, device)
        else:
            dirs_cam = torch.stack([
                (xx - cx) / fx,
                (yy - cy) / fy,
                torch.ones_like(xx)
            ], dim=-1)
        
        rays_d = F.normalize(dirs_cam @ R.T, dim=-1)
        rays_o = t.view(1, 1, 3).expand(H, W, 3)
        
        return rays_o.contiguous(), rays_d.contiguous()

    def _fisheye_unproject(self, xx, yy, fx, fy, cx, cy, camera, device):
        """Kannala-Brandt 鱼眼反投影."""
        if self.config.fisheye_distortion is not None:
            k = torch.tensor(self.config.fisheye_distortion, device=device, dtype=torch.float32)
        elif hasattr(camera, 'distortion_params'):
            k = camera.distortion_params.to(device).float()
        else:
            raise ValueError("Fisheye requires distortion_params")
        
        if len(k) < 4:
            k = torch.cat([k, torch.zeros(4 - len(k), device=device)])
        k = k[:4]
        
        u, v = (xx - cx) / fx, (yy - cy) / fy
        rho = torch.sqrt(u*u + v*v)
        phi = torch.atan2(v, u)
        
        theta = rho.clone()
        for _ in range(4):
            t2, t4, t6, t8 = theta**2, theta**4, theta**6, theta**8
            poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
            f = theta * poly - rho
            fp = poly + theta * (2*k[0]*theta + 4*k[1]*t2 + 6*k[2]*t4 + 8*k[3]*t6)
            fp = torch.where(fp.abs() > 1e-6, fp, torch.ones_like(fp) * 1e-6)
            theta = theta - f / fp
        
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)
        
        return torch.stack([sin_t * cos_p, sin_t * sin_p, cos_t], dim=-1)

    def _build_gauss_params_with_sh(self) -> dict:
        """构建 gauss_params（用独立 SH 替换 features）."""
        gauss_params_sh = {}
        
        # Linus: 精确排除，不用模糊匹配！
        EXCLUDE_KEYS = {'features_dc', 'features_rest'}
        
        for key, val in self.gauss_params.items():
            if key not in EXCLUDE_KEYS:  # ← 精确匹配
                gauss_params_sh[key] = val
        
        # 添加独立 SH
        gauss_params_sh['features_dc'] = self.sh_coeffs_dc
        gauss_params_sh['features_rest'] = self.sh_coeffs_rest
        
        return gauss_params_sh

    def _postprocess_outputs(self, outputs: dict, H: int, W: int) -> Dict[str, Tensor]:
        """后处理."""
        rgb, alpha, depth = outputs['rgb'], outputs['alpha'], outputs['depth']
        
        alpha = alpha.unsqueeze(-1) if alpha.ndim == 2 else alpha
        depth = depth.squeeze(-1) if depth.ndim == 3 else depth
        
        bg = self._get_background_color()
        if bg.ndim == 1:
            bg = bg.view(1, 1, 3).expand(H, W, 3)
        
        rgb = torch.clamp(rgb * alpha + (1 - alpha) * bg, 0, 1)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": alpha.squeeze(-1),
            "background": bg,
        }

    def get_param_groups(self) -> Dict[str, list]:
        """优化器参数组."""
        groups = super().get_param_groups()
        
        # 移除父类添加的 features 组（LiDAR 用的）
        # 因为我们用独立的 SH 替代
        if 'features_dc' in groups:
            del groups['features_dc']
        if 'features_rest' in groups:
            del groups['features_rest']
        
        # 添加独立 SH 组
        if hasattr(self, 'sh_coeffs_dc') and hasattr(self, 'sh_coeffs_rest'):
            groups['sh'] = [self.sh_coeffs_dc, self.sh_coeffs_rest]
        
        return groups

    def _get_background_color(self) -> Tensor:
        if hasattr(super(), '_get_background_color'):
            return super()._get_background_color()
        return torch.ones(3, device=self.sh_coeffs_dc.device)

    def _get_downscale_factor(self) -> float:
        if hasattr(super(), '_get_downscale_factor'):
            return super()._get_downscale_factor()
        return 4.0