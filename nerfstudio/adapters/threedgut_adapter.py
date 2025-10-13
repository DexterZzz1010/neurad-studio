"""
3DGUT Adapter - Pure SH rendering interface.

Design principles:
    1. Single responsibility: Convert SplatAD → 3DGUT format
    2. Fail-fast: Validate everything at init
    3. No special cases: One normalization, one validation path
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
    """Convert camera parameters to 3DGUT Batch format."""

    @staticmethod
    def convert(camera, c2w: Tensor, rays_o: Tensor, rays_d: Tensor,
                camera_model: str = "pinhole", 
                distortion: Optional[np.ndarray] = None) -> Batch:
        H, W = rays_o.shape[:2]
        c2w = c2w[0] if c2w.dim() == 3 else c2w

        fx, fy = float(camera.fx.item()), float(camera.fy.item())
        cx, cy = float(camera.cx.item()), float(camera.cy.item())

        if camera_model == "fisheye":
            if distortion is None:
                distortion = getattr(camera, 'distortion_params', None)
                if distortion is None:
                    raise ValueError("Fisheye requires distortion_params [k1,k2,k3,k4]")
                if isinstance(distortion, Tensor):
                    distortion = distortion.cpu().numpy()
            
            distortion = np.asarray(distortion, dtype=np.float32)
            if len(distortion) < 4:
                distortion = np.pad(distortion, (0, 4 - len(distortion)))
            
            max_angle = CamerasToBatchConverter._compute_max_angle(fx, fy, cx, cy, W, H)
            
            params = OpenCVFisheyeCameraModelParameters(
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fx, fy], dtype=np.float32),
                radial_coeffs=distortion[:4],
                resolution=np.array([W, H], dtype=np.int64),
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            key = "intrinsics_OpenCVFisheyeCameraModelParameters"
        else:
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([W, H], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fx, fy], dtype=np.float32),
                radial_coeffs=np.zeros(6, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            )
            key = "intrinsics_OpenCVPinholeCameraModelParameters"

        return Batch(**{
            'rays_ori': rays_o.unsqueeze(0).contiguous(),
            'rays_dir': rays_d.unsqueeze(0).contiguous(),
            'T_to_world': c2w.unsqueeze(0).contiguous(),
            key: params.to_dict(),
        })

    @staticmethod
    def _compute_max_angle(fx, fy, cx, cy, W, H):
        max_r = np.sqrt(max(cx**2 + cy**2, (W-cx)**2 + cy**2,
                           cx**2 + (H-cy)**2, (W-cx)**2 + (H-cy)**2))
        return float(max(2.0 * max_r / fx, 2.0 * max_r / fy) / 2.0)


class GaussiansWrapper:
    """Wrap SplatAD Gaussians for 3DGUT."""

    def __init__(self, model, target_radius: float, gut_sh_degree: int):
        g = model.gaussians
        dtype = torch.float32
        
        # Positions: normalize to sphere
        self.positions = self._normalize_positions(g.means.to(dtype), target_radius)
        
        # Rotations: unit quaternions
        self.quats = F.normalize(g.quats.to(dtype), dim=-1).contiguous()
        
        # Scales: ensure positive
        scales = g.scales.to(dtype)
        if torch.any(scales < 0):
            scales = torch.exp(scales)
        self.scales = torch.clamp(scales, min=1e-6).contiguous()
        
        # Opacities: map to [0,1]
        op = g.opacities.to(dtype)
        if op.min() < 0 or op.max() > 1:
            op = torch.sigmoid(op)
        self.opacity = torch.clamp(op, 0, 1).contiguous()
        
        # SH: map degree
        dc = g.features_dc.to(dtype)
        rest = g.features_rest.to(dtype)
        L_src = self._infer_L(rest)
        self.sh = self._map_sh(dc, rest, L_src, gut_sh_degree)
        
        # Validate (fail-fast)
        self._validate()
        
        self.num_gaussians = self.positions.shape[0]

    @staticmethod
    def _normalize_positions(means: Tensor, target_radius: float) -> Tensor:
        center = means.mean(dim=0, keepdim=True)
        centered = means - center
        
        dists = torch.linalg.norm(centered, dim=-1)
        radius = torch.quantile(dists, 0.99).item()
        radius = max(radius, centered.abs().max().item() * 0.5, 1e-6)
        
        normalized = centered * (target_radius / radius)
        normalized = torch.clamp(normalized, -2*target_radius, 2*target_radius)
        
        return normalized.contiguous()

    @staticmethod
    def _infer_L(features_rest: Tensor) -> int:
        M = features_rest.shape[-1] + 1
        L = int(M ** 0.5) - 1
        if (L + 1) ** 2 != M:
            raise ValueError(f"features_rest dim {M-1} invalid")
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
    """Thin wrapper around threedgut_tracer.Tracer."""

    def __init__(self, config: dict):
        if not THREEDGUT_AVAILABLE:
            raise ImportError("threedgut not available. Install: pip install threedgut")
        
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))

    def render(self, model, camera, rays_o: Tensor, rays_d: Tensor, c2w: Tensor,
               target_radius: float, gut_sh_degree: int) -> Dict[str, Tensor]:
        # Wrap Gaussians
        gaussians = GaussiansWrapper(model, target_radius, gut_sh_degree)
        
        # Get distortion if fisheye
        distortion = None
        if hasattr(model.config, 'fisheye_distortion'):
            dist = model.config.fisheye_distortion
            if dist is not None:
                distortion = np.array(dist, dtype=np.float32)
        
        # Convert camera
        camera_model = getattr(model.config, "camera_model", "pinhole")
        batch = CamerasToBatchConverter.convert(
            camera, c2w, rays_o, rays_d, camera_model, distortion
        )
        
        # Render
        outputs = self.tracer.render(gaussians, batch, train=model.training)
        
        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }