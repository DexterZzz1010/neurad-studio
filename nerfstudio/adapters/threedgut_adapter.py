"""
3DGUT Adapter - SplatAD → 3DGUT 转换层

设计原则（Linus style - COMPLETE）:
    1. Single responsibility: 只做格式转换
    2. Complete interface: 实现3DGUT的ExportableModel协议
    3. Fail-fast: 在 __init__ 就校验所有参数
    4. No special cases: 统一的getter模式，消除if分支
"""

import numpy as np
from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import threedgut_tracer
    from threedgrut.datasets.protocols import Batch
    from threedgrut.datasets.camera_models import (
        ShutterType,
        OpenCVPinholeCameraModelParameters,
        OpenCVFisheyeCameraModelParameters,
    )
    THREEDGUT_AVAILABLE = True
except ImportError:
    THREEDGUT_AVAILABLE = False

class _PassthroughBackground(torch.nn.Module):
    """
    与 3DGUT background 接口兼容的最小实现。
    作用：不做任何合成，直接返回 tracer 的输出（相当于透明背景/黑底）。
    签名需要与 tracer 调用保持一致：
        (T_to_world, rays_d, pred_rgb, pred_opacity, train) -> (pred_rgb, pred_opacity)
    """
    def forward(self, T_to_world, rays_d, pred_rgb, pred_opacity, train: bool):
        # pred_rgb: [1, H, W, 3], pred_opacity: [1, H, W, 1]
        return pred_rgb, pred_opacity

class CamerasToBatchConverter:
    """相机参数 → 3DGUT Batch."""

    @staticmethod
    def convert(camera, c2w: Tensor, rays_o: Tensor, rays_d: Tensor,
                camera_model: str = "pinhole", 
                distortion: Optional[np.ndarray] = None) -> Batch:
        H, W = rays_o.shape[:2]
        c2w = c2w[0] if c2w.dim() == 3 else c2w

        fx, fy = float(camera.fx.item()), float(camera.fy.item())
        cx, cy = float(camera.cx.item()), float(camera.cy.item())

        if camera_model == "fisheye":
            params, key = CamerasToBatchConverter._build_fisheye(
                fx, fy, cx, cy, W, H, camera, distortion
            )
        else:
            params, key = CamerasToBatchConverter._build_pinhole(
                fx, fy, cx, cy, W, H
            )

        return Batch(**{
            'rays_ori': rays_o.unsqueeze(0).contiguous(),
            'rays_dir': rays_d.unsqueeze(0).contiguous(),
            'T_to_world': c2w.unsqueeze(0).contiguous(),
            key: params.to_dict(),
        })

    @staticmethod
    def _build_fisheye(fx, fy, cx, cy, W, H, camera, distortion):
        if distortion is None:
            distortion = getattr(camera, 'distortion_params', None)
            if distortion is None:
                raise ValueError("Fisheye requires distortion_params [k1,k2,k3,k4]")
            if isinstance(distortion, Tensor):
                distortion = distortion.cpu().numpy()
        
        distortion = np.asarray(distortion, dtype=np.float32)
        if len(distortion) < 4:
            distortion = np.pad(distortion, (0, 4 - len(distortion)))
        
        max_r = np.sqrt(max(cx**2 + cy**2, (W-cx)**2 + cy**2,
                           cx**2 + (H-cy)**2, (W-cx)**2 + (H-cy)**2))
        max_angle = float(max(2.0 * max_r / fx, 2.0 * max_r / fy) / 2.0)
        
        params = OpenCVFisheyeCameraModelParameters(
            principal_point=np.array([cx, cy], dtype=np.float32),
            focal_length=np.array([fx, fy], dtype=np.float32),
            radial_coeffs=distortion[:4],
            resolution=np.array([W, H], dtype=np.int64),
            max_angle=max_angle,
            shutter_type=ShutterType.GLOBAL,
        )
        return params, "intrinsics_OpenCVFisheyeCameraModelParameters"

    @staticmethod
    def _build_pinhole(fx, fy, cx, cy, W, H):
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([W, H], dtype=np.int64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cx, cy], dtype=np.float32),
            focal_length=np.array([fx, fy], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        return params, "intrinsics_OpenCVPinholeCameraModelParameters"


class GaussiansWrapper:
    """
    从 SplatAD gauss_params 字典创建 3DGUT Gaussians.
    
    实现 3DGUT 的 ExportableModel 协议！
    
    Linus 好品味：
        - 属性命名匹配3DGUT（rotation不是quats，scale不是scales）
        - 统一getter接口，消除特殊情况
        - 默认返回激活后的值（99%的用例）
    """

    def __init__(self, gauss_params: dict, target_radius: float, gut_sh_degree: int):
        """
        Args:
            gauss_params: SplatAD 的 gauss_params 字典
            target_radius: 归一化半径
            gut_sh_degree: 目标 SH 度数
        """
        # 从字典提取参数
        means = self._get_param(gauss_params, 'means')
        quats = self._get_param(gauss_params, 'quats')
        scales = self._get_param(gauss_params, 'scales')
        opacities = self._get_param(gauss_params, 'opacities')
        features_dc = self._get_param(gauss_params, 'features_dc')
        features_rest = self._get_param(gauss_params, 'features_rest')
        
        dtype = torch.float32
        N = means.shape[0]
        
        # ============================================================
        # 核心属性：使用3DGUT的命名约定！
        # ============================================================
        
        # 位置归一化
        self.positions = self._normalize_positions(means.to(dtype), target_radius)
        
        # 旋转：3DGUT用 'rotation' 不是 'quats'
        # 已经是归一化的四元数
        self.rotation = F.normalize(quats.to(dtype), dim=-1).contiguous()
        
        # 尺度：3DGUT用 'scale' 不是 'scales'
        # 确保正数（SplatAD可能用log空间）
        scales_f = scales.to(dtype)
        if torch.any(scales_f < 0):
            scales_f = torch.exp(scales_f)
        self.scale = torch.clamp(scales_f, min=1e-6).contiguous()
        
        # 密度：3DGUT用 'density' 不是 'opacity'
        # 从opacity [0,1] 映射到density（logit空间）
        op = opacities.to(dtype)
        if op.min() < 0 or op.max() > 1:
            op = torch.sigmoid(op)
        op = torch.clamp(op, 1e-6, 1 - 1e-6)  # 避免log(0)
        self.density = torch.logit(op).reshape(-1, 1).contiguous()  # [N, 1]
        
        # SH特征：分成albedo和specular
        dc = features_dc.to(dtype)  # [N, 3, 1]
        rest = features_rest.to(dtype)  # [N, 3, M-1]
        L_src = self._infer_L(rest)
        sh_full = self._map_sh(dc, rest, L_src, gut_sh_degree)  # [N, 3, M]
        
        # 拆分成albedo（第0阶）和specular（高阶）
        self.features_albedo = sh_full[..., 0].contiguous()  # [N, 3]
        self.features_specular = sh_full[..., 1:].reshape(N, -1).contiguous()  # [N, 3*(M-1)]
        
        # ============================================================
        # 元数据属性
        # ============================================================
        self.max_n_features = gut_sh_degree
        self.n_active_features = (gut_sh_degree + 1) ** 2
        self.num_gaussians = N
        self.background = _PassthroughBackground().to(self.positions.device)
        
        # Linus: Fail-fast！
        self._validate()

    # ============================================================
    # 3DGUT ExportableModel 协议：必需的 getter 方法
    # ============================================================
    
    def get_positions(self) -> Tensor:
        """返回3D位置 [N, 3]."""
        return self.positions
    
    def get_rotation(self, preactivation: bool = False) -> Tensor:
        """
        返回旋转四元数 [N, 4].
        
        Args:
            preactivation: 忽略（四元数没有激活函数）
        """
        return self.rotation
    
    def get_scale(self, preactivation: bool = False) -> Tensor:
        """
        返回各向异性尺度 [N, 3].
        
        Args:
            preactivation: True=返回log空间, False=返回真实尺度（默认）
        """
        if preactivation:
            # 返回log空间（用于导出PLY）
            return torch.log(self.scale)
        return self.scale
    
    def get_density(self, preactivation: bool = False) -> Tensor:
        """
        返回密度 [N, 1].
        
        Args:
            preactivation: True=返回logit空间, False=返回sigmoid后的值（默认）
        """
        if preactivation:
            return self.density
        return torch.sigmoid(self.density)
    
    def get_features_albedo(self) -> Tensor:
        """返回0阶SH系数 [N, 3]."""
        return self.features_albedo
    
    def get_features_specular(self) -> Tensor:
        """返回高阶SH系数 [N, 3*(M-1)]."""
        return self.features_specular
    
    def get_features(self) -> Tensor:
        """返回完整特征（合并albedo和specular）."""
        return torch.cat([self.features_albedo, self.features_specular], dim=1)
    
    def get_max_n_features(self) -> int:
        """返回最大SH度数."""
        return self.max_n_features
    
    def get_n_active_features(self) -> int:
        """返回当前激活的SH系数数量."""
        return self.n_active_features
    
    # ============================================================
    # 辅助方法（私有）
    # ============================================================
    
    @staticmethod
    def _get_param(params_dict: dict, key_fragment: str) -> Tensor:
        """
        从字典中查找包含 key_fragment 的参数.
        
        Linus: 不要猜测，找不到就立即崩溃！
        """
        for k, v in params_dict.items():
            if key_fragment.lower() in k.lower():
                return v
        raise KeyError(
            f"WTF: Cannot find '{key_fragment}' in gauss_params!\n"
            f"Available keys: {list(params_dict.keys())}"
        )

    @staticmethod
    def _normalize_positions(means: Tensor, target_radius: float) -> Tensor:
        """
        位置归一化：中心化并缩放到目标半径.
        
        Linus: 简单直接，不要过度设计！
        """
        center = means.mean(dim=0, keepdim=True)
        centered = means - center
        
        # 用99%分位数估计半径（鲁棒于离群点）
        dists = torch.linalg.norm(centered, dim=-1)
        radius = torch.quantile(dists, 0.99).item()
        radius = max(radius, centered.abs().max().item() * 0.5, 1e-6)
        
        normalized = centered * (target_radius / radius)
        return torch.clamp(normalized, -2*target_radius, 2*target_radius).contiguous()

    @staticmethod
    def _infer_L(features_rest: Tensor) -> int:
        """
        从 features_rest 的形状推断SH度数L.
        
        features_rest: [N, 3, M-1] 其中 M = (L+1)^2
        """
        M = features_rest.shape[-1] + 1
        L = int(M ** 0.5) - 1
        if (L + 1) ** 2 != M:
            raise ValueError(
                f"WTF: features_rest dim {M-1} invalid!\n"
                f"Expected (L+1)^2 - 1 for some integer L"
            )
        return L

    @staticmethod
    def _map_sh(dc: Tensor, rest: Tensor, L_src: int, L_tgt: int) -> Tensor:
        """
        映射SH系数：截断或填充到目标度数.
        
        Linus: 消除特殊情况的经典案例！
                不需要if L_src > L_tgt ... elif L_src < L_tgt ...
                统一处理：先cat，然后切片或填充。
        """
        M_src, M_tgt = (L_src + 1) ** 2, (L_tgt + 1) ** 2
        sh = torch.cat([dc, rest], dim=-1)  # [N, 3, M_src]
        
        if M_src >= M_tgt:
            # 截断：直接切片
            return sh[..., :M_tgt].contiguous()
        else:
            # 填充：补零
            pad = torch.zeros(
                sh.shape[0], 3, M_tgt - M_src,
                device=sh.device, dtype=sh.dtype
            )
            return torch.cat([sh, pad], dim=-1).contiguous()

    def _validate(self):
        """
        Linus 风格：早检查，早崩溃！
        
        如果这个函数通过了，tracer就没有任何借口崩溃！
        """
        N = self.num_gaussians
        specular_dim = 3 * (self.n_active_features - 1)  # 3 * (M - 1)
        
        # 检查所有必需属性存在
        required = [
            'positions', 'rotation', 'scale', 'density',
            'features_albedo', 'features_specular',
            'n_active_features', 'max_n_features', 'num_gaussians'
        ]
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"WTF: Missing required attribute '{attr}'!\n"
                    f"This is a BUG in GaussiansWrapper!"
                )
        
        # 检查形状（用字典避免重复代码）
        specs = {
            'positions': (self.positions, (N, 3)),
            'rotation': (self.rotation, (N, 4)),
            'scale': (self.scale, (N, 3)),
            'density': (self.density, (N, 1)),
            'features_albedo': (self.features_albedo, (N, 3)),
            'features_specular': (self.features_specular, (N, specular_dim)),
        }
        
        for name, (tensor, expected_shape) in specs.items():
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"WTF: {name} has wrong shape!\n"
                    f"Expected: {expected_shape}\n"
                    f"Got: {tensor.shape}"
                )
            
            if not tensor.is_cuda:
                raise RuntimeError(f"WTF: {name} not on CUDA!")
            
            if tensor.dtype != torch.float32:
                raise TypeError(f"WTF: {name} not float32, got {tensor.dtype}")
            
            if not tensor.is_contiguous():
                raise RuntimeError(f"WTF: {name} not contiguous!")
            
            if not torch.isfinite(tensor).all():
                raise ValueError(f"WTF: {name} has NaN or Inf!")
        
        # 检查元数据一致性
        if self.n_active_features != (self.max_n_features + 1) ** 2:
            raise ValueError(
                f"WTF: n_active_features={self.n_active_features} "
                f"but max_n_features={self.max_n_features} "
                f"(expected {(self.max_n_features + 1) ** 2})"
            )
        
        # 检查四元数已归一化
        quat_norms = torch.linalg.norm(self.rotation, dim=-1)
        if not torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-4):
            raise ValueError(
                f"WTF: rotation quaternions not normalized!\n"
                f"Norm range: [{quat_norms.min():.6f}, {quat_norms.max():.6f}]"
            )
        
        print(f"[GaussiansWrapper] ✓ Validation passed: {N} gaussians, SH degree {self.max_n_features}")


class GUT3DRenderer:
    """
    3DGUT 渲染器封装.
    
    Linus: 简单封装，不要增加不必要的复杂度！
    """

    def __init__(self, config: dict):
        if not THREEDGUT_AVAILABLE:
            raise ImportError(
                "WTF: threedgut not available!\n"
                "Install it first or check your environment"
            )
        
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))
        print("[GUT3DRenderer] ✓ Initialized")

    def render(
        self, 
        gauss_params: dict, 
        camera, 
        rays_o: Tensor, 
        rays_d: Tensor, 
        c2w: Tensor, 
        target_radius: float, 
        gut_sh_degree: int,
        camera_model: str = "pinhole", 
        distortion: Optional[np.ndarray] = None, 
        training: bool = True
    ) -> Dict[str, Tensor]:
        """
        渲染接口.
        
        Args:
            gauss_params: SplatAD 的 gauss_params 字典
            camera: nerfstudio Camera 对象
            rays_o, rays_d: [H, W, 3] 射线原点和方向
            c2w: [3, 4] 或 [1, 3, 4] 相机到世界变换
            target_radius: 归一化半径
            gut_sh_degree: SH 度数
            camera_model: 'pinhole' 或 'fisheye'
            distortion: 鱼眼畸变系数（仅fisheye需要）
            training: 训练模式
            
        Returns:
            {'rgb': [H, W, 3], 'depth': [H, W, 1], 'alpha': [H, W, 1]}
        """
        # 包装 Gaussians（会自动校验）
        gaussians = GaussiansWrapper(gauss_params, target_radius, gut_sh_degree)
        
        # 转换相机参数到 Batch
        batch = CamerasToBatchConverter.convert(
            camera, c2w, rays_o, rays_d, camera_model, distortion
        )
        
        # 渲染
        try:
            outputs = self.tracer.render(gaussians, batch, train=training)
        except Exception as e:
            raise RuntimeError(
                f"WTF: 3DGUT tracer.render() failed!\n"
                f"Error: {e}\n"
                f"Gaussians: {gaussians.num_gaussians} points\n"
                f"Batch rays: {batch.rays_ori.shape}\n"
                f"Check 3DGUT tracer logs above for details"
            ) from e
        
        # 返回标准格式
        return {
            'rgb': outputs['pred_rgb'].squeeze(0),      # [H, W, 3]
            'depth': outputs['pred_dist'].squeeze(0),   # [H, W, 1]
            'alpha': outputs['pred_opacity'].squeeze(0), # [H, W, 1]
        }