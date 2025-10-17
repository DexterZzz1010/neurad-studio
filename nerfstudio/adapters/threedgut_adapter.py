"""
3DGUT Adapter for SplatAD - Feature→SH projection (adaptor-only).

Single responsibility:
- Convert SplatAD data to 3DGUT format (Batch + Gaussians)
- Map flat features (3 + feature_dim) → RGB×SH [3*(L+1)^2] WITHOUT adding trainable params
- Delegate to threedgut_tracer.Tracer.render()
"""

import hashlib
import math
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
    print("Warning: threedgut not available")


class CamerasToBatchConverter:
    """Convert nerfstudio Cameras to 3DGUT Batch."""

    @staticmethod
    def convert(camera, camera_to_world: Tensor, rays_o: Tensor, rays_d: Tensor,
                camera_model: str = "pinhole", fisheye_distortion: Optional[np.ndarray] = None) -> Batch:
        H, W = rays_o.shape[:2]
        c2w = camera_to_world[0] if camera_to_world.dim() == 3 else camera_to_world

        fx = float(camera.fx.item()); fy = float(camera.fy.item())
        cx = float(camera.cx.item()); cy = float(camera.cy.item())

        if camera_model == "auto":
            dist_params = getattr(camera, 'distortion_params', None)
            if dist_params is not None and not (dist_params == 0).all():
                camera_model = "fisheye"
                fisheye_distortion = dist_params.detach().cpu().numpy() if isinstance(dist_params, Tensor) else dist_params
            else:
                camera_model = "pinhole"

        if camera_model == "fisheye":
            if fisheye_distortion is None:
                dist_params = getattr(camera, 'distortion_params', None)
                if dist_params is None:
                    raise ValueError("Fisheye model requires distortion_params.")
                fisheye_distortion = dist_params.detach().cpu().numpy() if isinstance(dist_params, Tensor) else dist_params
            fisheye_distortion = np.asarray(fisheye_distortion, dtype=np.float32)
            if fisheye_distortion.shape[0] < 4:
                fisheye_distortion = np.pad(fisheye_distortion, (0, 4 - fisheye_distortion.shape[0]))
            fisheye_distortion = fisheye_distortion[:4].astype(np.float32)

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fx, fy], dtype=np.float32),
                radial_coeffs=fisheye_distortion,
                resolution=np.array([W, H], dtype=np.int64),
                max_angle=CamerasToBatchConverter._compute_fisheye_max_angle(fx, fy, cx, cy, W, H),
                shutter_type=ShutterType.GLOBAL,
            )
            intrinsics_key = "intrinsics_OpenCVFisheyeCameraModelParameters"
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
            intrinsics_key = "intrinsics_OpenCVPinholeCameraModelParameters"

        return Batch(**{
            'rays_ori': rays_o.unsqueeze(0).contiguous(),
            'rays_dir': rays_d.unsqueeze(0).contiguous(),
            'T_to_world': c2w.unsqueeze(0).contiguous(),
            intrinsics_key: params.to_dict(),
        })

    @staticmethod
    def _compute_fisheye_max_angle(fx, fy, cx, cy, W, H):
        max_radius = np.sqrt(max(
            cx**2 + cy**2, (W - cx)**2 + cy**2,
            cx**2 + (H - cy)**2, (W - cx)**2 + (H - cy)**2,
        ))
        return float(max(2.0 * max_radius / fx, 2.0 * max_radius / fy) / 2.0)


class GaussiansWrapper:
    """
    3DGUT Gaussians wrapper with optimized caching.
    """
    _projection_matrix_cache = {}

    def __init__(self, model, camera_extent=None):
        self.model = model
        device = model.means.device
        dtype = torch.float32

        # ============ POSITIONS: normalize ============
        cfg = getattr(model, "config", None)
        target_radius = float(getattr(cfg, "target_radius", 200.0))
        
        raw_means = model.means
        if not raw_means.is_cuda or raw_means.dtype != dtype:
            raw_means = raw_means.to(device=device, dtype=dtype)
        
        self.positions, self.normalization_center, self.normalization_scale = \
            self._normalize_positions_to_radius(raw_means, target_radius)
        
        assert self.positions.device == device and self.positions.dtype == dtype
        assert self.positions.is_contiguous()
        self.num_gaussians = self.positions.shape[0]

        # ============ ROTATION/SCALE/DENSITY ============
        rot = model.quats.to(dtype=dtype, device=device)
        rot = F.normalize(rot, dim=-1)
        assert torch.isfinite(rot).all() and (rot.norm(dim=-1) > 0.99).all()
        self.rotation = rot.contiguous()

        log_scale = model.scales.to(dtype=dtype, device=device)
        log_scale_clamped = torch.clamp(log_scale, min=-10, max=5)
        scl = torch.exp(log_scale_clamped)
        assert torch.isfinite(scl).all() and (scl > 0).all()
        self.scale = scl.contiguous()

        logit_opa = model.opacities.to(dtype=dtype, device=device)
        den = torch.sigmoid(logit_opa)
        if den.dim() == 1:
            den = den.unsqueeze(-1)
        assert den.shape == (self.num_gaussians, 1)
        assert torch.isfinite(den).all()
        self.density = den.contiguous()

        self.rotation_activation = lambda x: x
        self.scale_activation = lambda x: x
        self.density_activation = lambda x: x

        # ============ FEATURES: FORCE DEGREE=3 TO MATCH COMPILED KERNEL ============
        if not hasattr(model, 'features_rest'):
            raise AttributeError("model.features_rest not found")
        
        raw_rest = model.features_rest
        actual_K = int(raw_rest.shape[1])
        
        COMPILED_DEGREE = 3
        self._sh_degree = COMPILED_DEGREE
        M = (COMPILED_DEGREE + 1) ** 2 - 1  # 15
        self.feature_width = 3 * (M + 1)    # 48
        self.n_active_features = (COMPILED_DEGREE + 1) ** 2  # 16

        # 使用类级别缓存获取投影矩阵
        cache_key = (actual_K, 3 * M, str(device))
        if cache_key not in GaussiansWrapper._projection_matrix_cache:
            W = self._build_deterministic_projection(actual_K, 3 * M, device=device, dtype=dtype)
            W *= (1.0 / max(1, actual_K))
            GaussiansWrapper._projection_matrix_cache[cache_key] = W
        
        self._W = GaussiansWrapper._projection_matrix_cache[cache_key]
        assert self._W.device == device, f"W device {self._W.device} != target {device}"

        # Build features [N, 48] with coefficient-first RGB interleaving
        dc = model.features_dc.to(dtype=dtype, device=device)
        rest = raw_rest.to(dtype=dtype, device=device)

        assert dc.shape == (self.num_gaussians, 3)
        assert rest.shape == (self.num_gaussians, actual_K)

        # Project: [N,K] @ [K,45] = [N,45], reshape to [N,3,15]
        proj = rest @ self._W
        proj = proj.view(-1, 3, M)
        
        # Interleave: DC + projected high-order coeffs
        r = torch.cat([dc[:, 0:1], proj[:, 0, :]], dim=1)  # [N, 16]
        g = torch.cat([dc[:, 1:2], proj[:, 1, :]], dim=1)  # [N, 16]
        b = torch.cat([dc[:, 2:3], proj[:, 2, :]], dim=1)  # [N, 16]
        
        # Coefficient-first interleaving: [N, 16, 3] -> [N, 48]
        interleaved = torch.stack([r, g, b], dim=-1)
        features = interleaved.reshape(-1, 3 * (M + 1))

        assert features.shape == (self.num_gaussians, self.feature_width)
        assert features.device == device and features.dtype == dtype
        assert features.is_contiguous()
        assert torch.isfinite(features).all()
        
        self.features = features

        try:
            if cfg is not None:
                cfg.particle_radiance_sph_degree = self._sh_degree
        except Exception:
            pass

    @staticmethod
    def _normalize_positions_to_radius(means: torch.Tensor, target_radius: float):
        """Normalize positions to fit within target_radius sphere."""
        assert means.is_cuda and means.dtype == torch.float32
        assert target_radius > 0

        center = means.mean(dim=0, keepdim=False)
        centered = means - center

        dists = torch.linalg.norm(centered, dim=-1)
        q99 = torch.quantile(dists, 0.99).item()
        linf = centered.abs().amax().item()
        
        radius = max(q99, linf * 0.5, 1e-6)
        scale_factor = target_radius / radius

        normalized = centered * scale_factor
        normalized = torch.clamp(normalized, min=-2.0 * target_radius, max=2.0 * target_radius)

        assert normalized.is_cuda and normalized.dtype == torch.float32
        assert torch.isfinite(normalized).all()

        return normalized.contiguous(), center.detach(), torch.tensor(scale_factor, device=means.device)

    def get_rotation(self): return self.rotation
    def get_scale(self): return self.scale
    def get_density(self): return self.density
    def get_features(self): return self.features

    def background(self, T_to_world, rays_d, pred_rgb, pred_opacity, train):
        return pred_rgb, pred_opacity

    @staticmethod
    def _build_deterministic_projection(rows: int, cols: int, *, device, dtype) -> torch.Tensor:
        """Create an orthogonal matrix that is deterministic given (rows, cols)."""
        seed_material = f"{rows}x{cols}".encode("utf-8")
        seed_bytes = hashlib.sha256(seed_material).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big")

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        base = torch.randn(rows, cols, generator=gen, device=device, dtype=dtype)

        if rows < cols:
            q, r = torch.linalg.qr(base.t(), mode='reduced')
            q = q.t()
        else:
            q, r = torch.linalg.qr(base, mode='reduced')

        diag = torch.diagonal(r)
        phase = torch.sign(diag)
        phase[phase == 0] = 1.0
        q = q * phase

        return q.contiguous()


class GUT3DRenderer:
    """Thin wrapper around threedgut_tracer.Tracer with caching."""

    def __init__(self, config: dict):
        if not THREEDGUT_AVAILABLE:
            raise ImportError("threedgut_tracer not available")
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))
        self._wrapper_cache = {}

    def render(self, model, camera, rays_o, rays_d, c2w) -> Dict[str, Tensor]:
        # 缓存策略：基于model id和数据指针
        model_id = id(model)
        cache_key = (
            model_id,
            model.means.data_ptr(),
            model.num_points if hasattr(model, 'num_points') else model.means.shape[0]
        )
        
        # 检查缓存是否有效
        if cache_key not in self._wrapper_cache:
            # 清理旧缓存避免内存泄漏
            if len(self._wrapper_cache) > 10:
                self._wrapper_cache.clear()
            
            gaussians = GaussiansWrapper(model)
            self._wrapper_cache[cache_key] = gaussians
        else:
            gaussians = self._wrapper_cache[cache_key]

        fisheye_dist = None
        if hasattr(model, 'config'):
            fisheye_dist = getattr(model.config, 'fisheye_distortion', None)
            if fisheye_dist is not None:
                fisheye_dist = np.array(fisheye_dist, dtype=np.float32)

        batch = CamerasToBatchConverter.convert(
            camera=camera,
            camera_to_world=c2w,
            rays_o=rays_o,
            rays_d=rays_d,
            camera_model=getattr(model.config, "camera_model", "pinhole"),
            fisheye_distortion=fisheye_dist,
        )

        try:
            outputs = self.tracer.render(gaussians, batch, train=model.training)
        except RuntimeError as e:
            # 如果失败，清除缓存并重试一次
            if "illegal memory access" in str(e):
                self._wrapper_cache.clear()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gaussians = GaussiansWrapper(model)
                outputs = self.tracer.render(gaussians, batch, train=model.training)
            else:
                raise

        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }
