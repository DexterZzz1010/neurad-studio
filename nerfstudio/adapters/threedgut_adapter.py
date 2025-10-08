"""
3DGUT Adapter for SplatAD - The Right Way.

Single responsibility: Convert SplatAD data to 3DGUT format and delegate to Tracer.
No rendering logic, no parameter conversion - those already exist in threedgut_tracer.

Design:
    - CamerasToBatchConverter: Cameras → Batch
    - GaussiansWrapper: SplatAD params → 3DGUT Gaussians interface
    - Call threedgut_tracer.Tracer.render() directly

Total: ~80 LOC vs previous 400 LOC garbage.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, Optional

# 3DGUT imports
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
    print("Warning: threedgut not available, ray tracing disabled")


class CamerasToBatchConverter:
    """
    Convert nerfstudio Cameras to 3DGUT Batch.

    Reuses ALL existing 3DGUT logic - we just prepare the data structure.
    """

    @staticmethod
    def convert(
        camera,
        camera_to_world: Tensor,  # [4, 4] or [1, 4, 4]
        rays_o: Tensor,           # [H, W, 3]
        rays_d: Tensor,           # [H, W, 3]
        camera_model: str = "pinhole",  # From config
        fisheye_distortion: Optional[np.ndarray] = None,  # From config or camera
    ) -> Batch:
        """
        Build Batch matching 3DGUT's __getitem__ output.
        Tracer.__create_camera_parameters() will handle the rest.
        """
        H, W = rays_o.shape[:2]
        c2w = camera_to_world[0] if camera_to_world.dim() == 3 else camera_to_world

        # Extract intrinsics
        fx = float(camera.fx.item())
        fy = float(camera.fy.item())
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())

        # Infer camera model if "auto"
        if camera_model == "auto":
            dist_params = getattr(camera, 'distortion_params', None)
            if dist_params is not None and not (dist_params == 0).all():
                camera_model = "fisheye"
                fisheye_distortion = dist_params.detach().cpu().numpy() if isinstance(dist_params, Tensor) else dist_params
            else:
                camera_model = "pinhole"

        # Build 3DGUT camera model parameters
        if camera_model == "fisheye":
            if fisheye_distortion is None:
                dist_params = getattr(camera, 'distortion_params', None)
                if dist_params is None:
                    raise ValueError("Fisheye model requires distortion_params.")
                fisheye_distortion = dist_params.detach().cpu().numpy() if isinstance(dist_params, Tensor) else dist_params

            # Ensure 4 coefficients [k1, k2, k3, k4]
            if len(fisheye_distortion) < 4:
                fisheye_distortion = np.pad(np.asarray(fisheye_distortion, dtype=np.float32), (0, 4 - len(fisheye_distortion)))
            else:
                fisheye_distortion = np.asarray(fisheye_distortion[:4], dtype=np.float32)

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

        # 3DGUT 期望 [1, H, W, 3]
        rays_o_b = rays_o.unsqueeze(0).contiguous()
        rays_d_b = rays_d.unsqueeze(0).contiguous()
        c2w_b = c2w.unsqueeze(0).contiguous()

        return Batch(**{
            'rays_ori': rays_o_b,
            'rays_dir': rays_d_b,
            'T_to_world': c2w_b,
            intrinsics_key: params.to_dict(),
        })

    @staticmethod
    def _compute_fisheye_max_angle(fx, fy, cx, cy, W, H):
        """From dataset_colmap.py - compute FOV proxy for fisheye."""
        max_radius = np.sqrt(max(
            cx**2 + cy**2,
            (W - cx)**2 + cy**2,
            cx**2 + (H - cy)**2,
            (W - cx)**2 + (H - cy)**2,
        ))
        return float(max(2.0 * max_radius / fx, 2.0 * max_radius / fy) / 2.0)


class GaussiansWrapper:
    """Wrap SplatAD parameters in 3DGUT's Gaussians interface with auto-normalization."""

    def __init__(self, model, camera_extent=None):
        self.model = model

        # ================== 空间尺度诊断 + 自动归一化 ==================
        means = model.means  # [N, 3]
        means_center = means.mean(dim=0)  # [3]
        means_std = means.std()  # scalar

        means_min = means.min()
        means_max = means.max()
        means_range = means_max - means_min

        print(f"\n{'='*70}")
        print(f"Gaussians Spatial Analysis")
        print(f"{'='*70}")
        print(f"means range: [{means_min.item():.2f}, {means_max.item():.2f}]")
        print(f"means center: {means_center.detach().cpu().numpy()}")
        print(f"means std: {means_std.item():.2f}")
        print(f"means span: {means_range.item():.2f}")

        REASONABLE_RANGE = 200.0
        if means_range > REASONABLE_RANGE:
            print(f"\n⚠️  WARNING: Gaussian positions out of reasonable range!")
            print(f"   Range {means_range.item():.2f} >> {REASONABLE_RANGE}")
            print(f"   Applying automatic normalization...")

            TARGET_SCALE = 50.0
            scale_factor = TARGET_SCALE / (3.0 * (means_std + 1e-12))

            print(f"   Normalization params:")
            print(f"     center: {means_center.detach().cpu().numpy()}")
            print(f"     scale_factor: {float(scale_factor):.6f}")

            normalized_means = (means - means_center) * scale_factor

            new_min = normalized_means.min()
            new_max = normalized_means.max()
            new_std = normalized_means.std()
            print(f"   After normalization:")
            print(f"     range: [{new_min.item():.2f}, {new_max.item():.2f}]")
            print(f"     std: {new_std.item():.2f}")
            print(f"{'='*70}\n")

            self.normalization_center = means_center
            self.normalization_scale = scale_factor
            self.positions = normalized_means.contiguous()
        else:
            print(f"✓ Gaussian positions in reasonable range, no normalization needed")
            print(f"{'='*70}\n")
            self.normalization_center = None
            self.normalization_scale = None
            self.positions = means.contiguous()

        self.num_gaussians = self.positions.shape[0]

        # ================== 特征通道构建（关键修复） ==================
        # 目标：最终 features 形状为 [N, 3*(L+1)^2]，且 n_active_features 与之相等
        N = self.model.features_dc.shape[0]
        dc = self.model.features_dc  # [N, 3]

        rest = getattr(self.model, "features_rest", None)
        rest_flat = None

        if rest is not None and rest.numel() > 0:
            if rest.dim() == 3 and rest.shape[-1] == 3:
                # Case B: [N, ((L+1)^2 - 1), 3] -> flatten to [N, ((L+1)^2 - 1)*3]
                rest_flat = rest.reshape(N, -1)
            elif rest.dim() == 2:
                # Case A: [N, K] — 已展平或灰度
                K = rest.shape[1]
                if K % 3 == 0:
                    # 已经是 RGB 展平的高阶项
                    rest_flat = rest
                else:
                    # 灰度 SH（((L+1)^2 - 1)），临时扩展到 RGB 以跑通
                    print("⚠️  features_rest appears grayscale; expanding to RGB by replication.")
                    rest_flat = rest.unsqueeze(-1).expand(-1, -1, 3).reshape(N, -1)
            else:
                raise RuntimeError(f"Unexpected features_rest shape: {tuple(rest.shape)}")
        else:
            # 仅 DC
            rest_flat = torch.empty((N, 0), device=dc.device, dtype=dc.dtype)

        features = torch.cat([dc, rest_flat], dim=1).contiguous()  # [N, F]
        F = features.shape[1]

        # 强校验：F 必须满足 3*(L+1)^2
        if F % 3 != 0:
            raise RuntimeError(f"Radiance features ({F}) must be multiple of 3 (RGB).")
        m = F // 3
        L_float = np.sqrt(m) - 1.0
        L = int(round(L_float))
        if (L + 1) ** 2 != m:
            raise RuntimeError(f"Radiance channels={F} invalid; expect 3*(L+1)^2. Got m={m}.")

        # 保存
        self._features = features
        self.n_active_features = F  # 关键：用拼接后的真实通道数

        # 额外：把推断出的 L 写回配置（若存在），避免跑偏
        cfg = getattr(self.model, "config", None)
        try:
            if cfg is not None and hasattr(cfg, "particle_radiance_sph_degree"):
                if getattr(cfg, "particle_radiance_sph_degree") != L:
                    print(f"ℹ️  Overriding config.particle_radiance_sph_degree -> {L}")
                    cfg.particle_radiance_sph_degree = L
        except Exception:
            pass

    # ---------- 3DGUT 访问接口 ----------
    def get_rotation(self):
        """Return normalized quaternions as [N, 4]."""
        quats = self.model.quats
        return quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)

    def get_scale(self):
        """Return activated scales as [N, 3] (exp)."""
        return torch.exp(self.model.scales)

    def get_density(self):
        """Return activated opacity as [N, 1] (sigmoid, column vector)."""
        opacities = torch.sigmoid(self.model.opacities)
        if opacities.dim() == 1:
            opacities = opacities.unsqueeze(-1)
        return opacities

    def get_features(self):
        """Return [N, 3*(L+1)^2] contiguous RGB SH features."""
        # 再次保底健诊
        feat = self._features
        assert feat.is_cuda and feat.dtype == torch.float32 and feat.is_contiguous()
        assert torch.isfinite(feat).all(), "features contain NaN/Inf"
        F = feat.shape[1]
        assert F == self.n_active_features, f"Feature mismatch: {F} vs {self.n_active_features}"
        return feat

    def background(self, T_to_world, rays_d, pred_rgb, pred_opacity, train):
        """Composite with background (optional, usually no-op)."""
        return pred_rgb, pred_opacity


class GUT3DRenderer:
    """
    Thin wrapper around threedgut_tracer.Tracer.

    Only job: Convert data and delegate to Tracer.render().
    """

    def __init__(self, config: dict):
        """Initialize 3DGUT Tracer with config."""
        if not THREEDGUT_AVAILABLE:
            raise ImportError("threedgut_tracer is not available.")
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))

    def render(self, model, camera, rays_o, rays_d, c2w) -> Dict[str, Tensor]:
        """
        Render using 3DGUT.

        Steps:
            1. Wrap Gaussians
            2. Build Batch (with camera model from config)
            3. Call Tracer.render()
            4. Done
        """

        # ================== 数据健康检查 ==================
        print("\n" + "="*70)
        print("PRE-RENDER DATA VALIDATION")
        print("="*70)

        def check_tensor(name, t):
            print(f"{name}:")
            print(f"  shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
            print(f"  contiguous={t.is_contiguous()}")
            t_min = float(torch.nan_to_num(t).min().item())
            t_max = float(torch.nan_to_num(t).max().item())
            print(f"  range=[{t_min:.4f}, {t_max:.4f}]")
            if torch.isnan(t).any():
                raise RuntimeError(f"NaN in {name}")
            if torch.isinf(t).any():
                raise RuntimeError(f"Inf in {name}")

        check_tensor("model.quats", model.quats)
        check_tensor("model.scales", model.scales)
        check_tensor("model.opacities", model.opacities)
        check_tensor("model.means", model.means)
        check_tensor("rays_o", rays_o)
        check_tensor("rays_d", rays_d)

        print("="*70 + "\n")

        # 1) Wrap Gaussians
        gaussians = GaussiansWrapper(model)

        # 2) Debug 输出一次形状概览
        if not hasattr(self, '_debug_printed'):
            print("\n" + "="*70)
            print("3DGUT Renderer - Data Shapes")
            print("="*70)
            print(f"Gaussians:")
            print(f"  num_gaussians: {gaussians.num_gaussians}")
            print(f"  positions: {gaussians.positions.shape} {gaussians.positions.dtype} {gaussians.positions.device}")
            print(f"  rotation: {gaussians.get_rotation().shape} {gaussians.get_rotation().dtype}")
            print(f"  scale: {gaussians.get_scale().shape} {gaussians.get_scale().dtype}")
            print(f"  density: {gaussians.get_density().shape} {gaussians.get_density().dtype}")
            print(f"  features: {gaussians.get_features().shape} {gaussians.get_features().dtype}")
            print(f"  n_active_features: {gaussians.n_active_features}")
            print(f"\nRays:")
            print(f"  rays_o: {rays_o.shape} {rays_o.dtype} {rays_o.device}")
            print(f"  rays_d: {rays_d.shape} {rays_d.dtype} {rays_d.device}")
            print(f"  c2w: {c2w.shape} {c2w.dtype}")
            print("="*70 + "\n")
            self._debug_printed = True

        # 3) Cameras → Batch
        fisheye_dist = None
        if hasattr(model, 'config') and hasattr(model.config, 'fisheye_distortion') and model.config.fisheye_distortion is not None:
            fisheye_dist = np.array(model.config.fisheye_distortion, dtype=np.float32)

        batch = CamerasToBatchConverter.convert(
            camera=camera,
            camera_to_world=c2w,
            rays_o=rays_o,
            rays_d=rays_d,
            camera_model=getattr(model.config, "camera_model", "pinhole"),
            fisheye_distortion=fisheye_dist,
        )

        # 4) 调用 Tracer.render
        try:
            outputs = self.tracer.render(gaussians, batch, train=model.training)
        except RuntimeError as e:
            print(f"\n[ERROR] 3DGUT render failed: {e}")
            print(f"Gaussians info:")
            print(f"  num: {gaussians.num_gaussians}")
            print(f"  n_active_features: {gaussians.n_active_features}")
            print(f"  All shapes:")
            print(f"    positions: {gaussians.positions.shape}")
            print(f"    rotation: {gaussians.get_rotation().shape}")
            print(f"    scale: {gaussians.get_scale().shape}")
            print(f"    density: {gaussians.get_density().shape}")
            print(f"    features: {gaussians.get_features().shape}")
            raise

        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }
