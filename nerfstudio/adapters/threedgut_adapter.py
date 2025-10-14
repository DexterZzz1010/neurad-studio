"""
3DGUT Adapter - SplatAD → 3DGUT 转换层

设计原则（Linus style）:
    1. Single responsibility: 只做格式转换
    2. Fail-fast: 在 __init__ 就校验所有参数
    3. No guessing: 显式传参，不从对象猜测
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
    
    关键：SplatAD 用 gauss_params 字典，不是 self.gaussians 对象！
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
        
        # 位置归一化
        self.positions = self._normalize_positions(means.to(dtype), target_radius)
        
        # 旋转归一化
        self.quats = F.normalize(quats.to(dtype), dim=-1).contiguous()
        
        # 尺度确保正数
        scales_f = scales.to(dtype)
        if torch.any(scales_f < 0):
            scales_f = torch.exp(scales_f)
        self.scales = torch.clamp(scales_f, min=1e-6).contiguous()
        
        # 不透明度映射到 [0,1]
        op = opacities.to(dtype)
        if op.min() < 0 or op.max() > 1:
            op = torch.sigmoid(op)
        self.opacity = torch.clamp(op, 0, 1).contiguous()
        
        # SH 度数映射
        dc = features_dc.to(dtype)
        rest = features_rest.to(dtype)
        L_src = self._infer_L(rest)
        self.sh = self._map_sh(dc, rest, L_src, gut_sh_degree)
        
        # Fail-fast 校验
        self._validate()
        
        self.num_gaussians = self.positions.shape[0]

    @staticmethod
    def _get_param(params_dict: dict, key_fragment: str) -> Tensor:
        """从字典中查找包含 key_fragment 的参数."""
        for k, v in params_dict.items():
            if key_fragment.lower() in k.lower():
                return v
        raise KeyError(f"Cannot find '{key_fragment}' in gauss_params keys: {list(params_dict.keys())}")

    @staticmethod
    def _normalize_positions(means: Tensor, target_radius: float) -> Tensor:
        center = means.mean(dim=0, keepdim=True)
        centered = means - center
        
        dists = torch.linalg.norm(centered, dim=-1)
        radius = torch.quantile(dists, 0.99).item()
        radius = max(radius, centered.abs().max().item() * 0.5, 1e-6)
        
        normalized = centered * (target_radius / radius)
        return torch.clamp(normalized, -2*target_radius, 2*target_radius).contiguous()

    @staticmethod
    def _infer_L(features_rest: Tensor) -> int:
        M = features_rest.shape[-1] + 1
        L = int(M ** 0.5) - 1
        if (L + 1) ** 2 != M:
            raise ValueError(f"features_rest dim {M-1} invalid (expected (L+1)^2-1)")
        return L

    @staticmethod
    def _map_sh(dc: Tensor, rest: Tensor, L_src: int, L_tgt: int) -> Tensor:
        M_src, M_tgt = (L_src + 1) ** 2, (L_tgt + 1) ** 2
        sh = torch.cat([dc, rest], dim=-1)
        
        if M_src == M_tgt:
            return sh.contiguous()
        elif M_src > M_tgt:
            return sh[..., :M_tgt].contiguous()
        else:
            pad = torch.zeros(sh.shape[0], 3, M_tgt - M_src, 
                            device=sh.device, dtype=sh.dtype)
            return torch.cat([sh, pad], dim=-1).contiguous()

    def _validate(self):
        N, M = self.positions.shape[0], self.sh.shape[-1]
        
        specs = {
            'positions': (self.positions, (N, 3)),
            'quats': (self.quats, (N, 4)),
            'scales': (self.scales, (N, 3)),
            'opacity': (self.opacity, (N, 1)),
            'sh': (self.sh, (N, 3, M)),
        }
        
        for name, (tensor, shape) in specs.items():
            assert tensor.shape == shape, f"{name}: {tensor.shape} != {shape}"
            assert tensor.is_cuda, f"{name} not CUDA"
            assert tensor.dtype == torch.float32, f"{name} not float32"
            assert tensor.is_contiguous(), f"{name} not contiguous"
            assert torch.isfinite(tensor).all(), f"{name} has NaN/Inf"


class GUT3DRenderer:
    """3DGUT 渲染器封装."""

    def __init__(self, config: dict):
        if not THREEDGUT_AVAILABLE:
            raise ImportError("threedgut not available")
        
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))

    def render(self, gauss_params: dict, camera, rays_o: Tensor, rays_d: Tensor, 
               c2w: Tensor, target_radius: float, gut_sh_degree: int,
               camera_model: str, distortion: Optional[np.ndarray], 
               training: bool) -> Dict[str, Tensor]:
        """
        渲染接口.
        
        Args:
            gauss_params: SplatAD 的 gauss_params 字典
            camera: nerfstudio Camera
            rays_o, rays_d: [H,W,3] 射线
            c2w: [3,4] 或 [1,3,4] 相机姿态
            target_radius: 归一化半径
            gut_sh_degree: SH 度数
            camera_model: 'pinhole' 或 'fisheye'
            distortion: 鱼眼畸变系数
            training: 训练模式
        """
        # 包装 Gaussians
        gaussians = GaussiansWrapper(gauss_params, target_radius, gut_sh_degree)
        
        # 转换相机
        batch = CamerasToBatchConverter.convert(
            camera, c2w, rays_o, rays_d, camera_model, distortion
        )
        
        # 渲染
        outputs = self.tracer.render(gaussians, batch, train=training)
        
        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }