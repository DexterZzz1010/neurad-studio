"""
SplatGUT: 相机用独立 SH，LiDAR 用 features + decoder

设计：
    1. Camera: 新 SH 参数 → 3DGUT → RGB (无 decoder)
    2. LiDAR: 原 features → decoder → intensity/ray_drop (继承父类)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig
from nerfstudio.cameras.cameras import Cameras


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
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
    config: SplatGUTModelConfig

    def populate_modules(self):
        """初始化：父类 → 创建独立 SH → 禁用相机 decoder → 初始化 3DGUT."""
        super().populate_modules()
        
        # 创建独立 SH（不动原 features）
        self._create_independent_sh()
        
        # 禁用相机 decoder（LiDAR decoder 保留）
        self._disable_camera_decoder()
        
        # 初始化 3DGUT
        if self.config.use_ray_tracing:
            self._init_gut_renderer()

    def _create_independent_sh(self):
        """创建独立 SH 参数（不影响 gauss_params 中的 features）."""
        # 获取 Gaussian 数量
        means = None
        for key, val in self.gauss_params.items():
            if 'means' in key.lower():
                means = val
                break
        
        if means is None:
            raise RuntimeError("Cannot find 'means' in gauss_params")
        
        N = means.shape[0]
        device = means.device
        dtype = means.dtype
        
        L = self.config.sh_degree
        M = (L + 1) ** 2
        
        # 创建独立 SH（RGB × SH 系数）
        self.sh_coeffs_dc = nn.Parameter(
            torch.zeros(N, 3, 1, device=device, dtype=dtype)
        )
        self.sh_coeffs_rest = nn.Parameter(
            torch.zeros(N, 3, M - 1, device=device, dtype=dtype)
        )
        
        print(f"[SplatGUT] Created independent SH: N={N}, L={L}, M={M}")
        print(f"  sh_coeffs_dc: {tuple(self.sh_coeffs_dc.shape)}")
        print(f"  sh_coeffs_rest: {tuple(self.sh_coeffs_rest.shape)}")
        print(f"  Original features_dc (for LiDAR): {self.gauss_params['features_dc'].shape}")
        print(f"  Original features_rest (for LiDAR): {self.gauss_params['features_rest'].shape}")

    def _disable_camera_decoder(self):
        """
        禁用相机 decoder（保留 LiDAR decoder）.
        
        注意：只禁用 rgb_decoder 和 appearance_embedding，
              lidar_decoder 保留用于 LiDAR 渲染。
        """
        if hasattr(self, 'rgb_decoder'):
            self.rgb_decoder = None
            print("[SplatGUT] Disabled rgb_decoder (camera will use SH)")
        
        if hasattr(self, 'appearance_embedding'):
            self.appearance_embedding = None
            print("[SplatGUT] Disabled appearance_embedding")
        
        # 保留 lidar_decoder
        if hasattr(self, 'lidar_decoder'):
            print("[SplatGUT] Kept lidar_decoder (LiDAR uses features)")

    def _init_gut_renderer(self):
        """初始化 3DGUT."""
        try:
            from nerfstudio.adapters.threedgut_adapter import GUT3DRenderer, THREEDGUT_AVAILABLE
        except ImportError as e:
            print(f"[SplatGUT] 3DGUT import failed: {e}")
            self.config.use_ray_tracing = False
            return
        
        if not THREEDGUT_AVAILABLE:
            print("[SplatGUT] 3DGUT not available")
            self.config.use_ray_tracing = False
            return
        
        L = self.config.sh_degree
        gut_config = {
            'render': {
                'method': '3dgut',
                'particle_radiance_sph_degree': L,
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
        print(f"[SplatGUT] 3DGUT ready: L={L}, k={self.config.k_buffer_size}")

    def get_param_groups(self) -> Dict[str, list]:
        """
        优化器参数组：移除原 features，添加独立 SH.
        
        关键：
            - 移除 features_dc/features_rest（它们现在只用于 LiDAR）
            - 添加 sh_coeffs_dc/sh_coeffs_rest（用于相机）
        """
        groups = super().get_param_groups()
        
        # 移除原来的 features_dc 和 features_rest
        # （它们在父类中可能被添加到 'sh' 或其他组）
        for group_name in list(groups.keys()):
            if group_name in ['features_dc', 'features_rest', 'sh']:
                # 过滤掉来自 gauss_params 的 features
                filtered = []
                for param in groups[group_name]:
                    param_id = id(param)
                    # 检查是否是 gauss_params 中的 features
                    is_gauss_feature = False
                    for key, val in self.gauss_params.items():
                        if 'features' in key.lower() and id(val) == param_id:
                            is_gauss_feature = True
                            break
                    
                    if not is_gauss_feature:
                        filtered.append(param)
                
                if filtered:
                    groups[group_name] = filtered
                else:
                    del groups[group_name]
        
        # 添加独立 SH
        sh_group = []
        if hasattr(self, 'sh_coeffs_dc'):
            sh_group.append(self.sh_coeffs_dc)
        if hasattr(self, 'sh_coeffs_rest'):
            sh_group.append(self.sh_coeffs_rest)
        
        if sh_group:
            groups["sh"] = sh_group
        
        return groups

    def get_outputs_for_camera(self, camera: Cameras) -> Dict[str, Tensor]:
        """相机渲染：使用独立 SH."""
        if not self.config.use_ray_tracing:
            return super().get_outputs_for_camera(camera)
        
        # 相机优化
        if self.training or self.config.use_camopt_in_eval:
            camera = self.camera_optimizer.apply_to_camera(camera)
        
        # 降采样
        scale = self._get_downscale_factor()
        if scale != 1:
            camera.rescale_output_resolution(1 / scale)
        
        W, H = int(camera.width.item()), int(camera.height.item())
        c2w = camera.camera_to_worlds
        
        # 生成射线
        rays_o, rays_d = self._generate_rays(camera, W, H, c2w)
        
        # 构建临时 gauss_params（用我们的 SH 替换 features）
        gauss_params_sh = self._build_gauss_params_with_sh()
        
        # 3DGUT 渲染
        distortion = None
        if self.config.fisheye_distortion is not None:
            distortion = torch.tensor(
                self.config.fisheye_distortion, 
                device=rays_o.device, 
                dtype=torch.float32
            ).cpu().numpy()
        
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
        
        # 合成背景
        rgb, alpha, depth = outputs['rgb'], outputs['alpha'], outputs['depth']
        
        alpha = alpha.unsqueeze(-1) if alpha.ndim == 2 else alpha
        depth = depth.squeeze(-1) if depth.ndim == 3 else depth
        
        bg = self._get_background_color()
        if bg.ndim == 1:
            bg = bg.view(1, 1, 3).expand(H, W, 3)
        
        rgb = torch.clamp(rgb * alpha + (1 - alpha) * bg, 0, 1)
        
        if scale != 1:
            camera.rescale_output_resolution(scale)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": alpha.squeeze(-1),
            "background": bg,
        }

    def _build_gauss_params_with_sh(self) -> dict:
        """构建临时 gauss_params（用 SH 替换 features）."""
        gauss_params_sh = {}
        
        # 复制所有参数（except features）
        for key, val in self.gauss_params.items():
            if 'features' not in key.lower():
                gauss_params_sh[key] = val
        
        # 添加我们的 SH（用 features 的键名，但值是 SH）
        gauss_params_sh['features_dc'] = self.sh_coeffs_dc
        gauss_params_sh['features_rest'] = self.sh_coeffs_rest
        
        return gauss_params_sh

    @torch.no_grad()
    def _generate_rays(self, camera: Cameras, W: int, H: int, c2w: Tensor):
        """生成射线（pinhole/fisheye）."""
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
        elif hasattr(camera, 'distortion_params') and camera.distortion_params is not None:
            k = camera.distortion_params.to(device).float()
        else:
            raise ValueError("Fisheye requires distortion_params [k1,k2,k3,k4]")
        
        if k.numel() < 4:
            k = torch.cat([k, torch.zeros(4 - k.numel(), device=device)])
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