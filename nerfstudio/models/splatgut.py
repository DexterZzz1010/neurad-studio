"""
SplatGUT model built on top of SplatAD with 3DGUT rasterization.

Key design:
- Storage: [N, feature_dim] flat format (same as SplatAD for LiDAR compatibility)
- Initialization: Semantically correct for SH (initialized from [N, K, 3])
- Camera rendering: Reshape to [N, K, 3] for gut_rasterization
- LiDAR rendering: Inherited from SplatAD, uses flat format directly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import warnings

import torch
from torch import nn
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig, random_quat_tensor
from nerfstudio.models.splatfacto import get_viewmat

try:
    from gsplat_original.cuda._wrapper import RollingShutterType
    from gsplat_original.rendering import rasterization as gut_rasterization
except ImportError as exc:
    raise ImportError(
        "Please install the gsplat_original package with 3DGUT support. "
        "See: https://github.com/nerfstudio-project/gsplat"
    ) from exc


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    """Configuration for the SplatGUT model."""

    _target: Type = field(default_factory=lambda: SplatGUTModel)

    use_gut_rasterization: bool = True
    """Use 3DGUT rasterization for camera rendering (with native SH support)."""

    with_ut: bool = True
    """Enable Unscented Transform projection for more accurate covariance."""

    with_eval3d: bool = True
    """Evaluate splats in 3D space (slower but more accurate)."""

    camera_model: Literal["pinhole", "fisheye", "ortho", "ftheta"] = "pinhole"
    """Camera model type passed to the 3DGUT rasterizer."""

    sh_degree: int = 3
    """Spherical harmonic degree for camera rendering."""

    # max_num_seed_points: int = 100_000_000
    # mcmc_cap_max: int = 200_000_000

    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""

class SplatGUTModel(SplatADModel):
    """SplatAD variant using 3DGUT rasterization for cameras with native SH support.
    
    Architecture:
    - Inherits all LiDAR functionality from SplatAD (unchanged)
    - Overrides camera rendering to use 3DGUT with proper SH format
    - Maintains storage format [N, feature_dim] for compatibility
    - Reshapes to [N, K, 3] only during camera forward pass
    """

    config: SplatGUTModelConfig

    def __init__(
        self,
        config: SplatGUTModelConfig,
        *args,
        seed_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        """Initialize SplatGUT model."""
        # Calculate SH coefficients for camera rendering
        self._num_sh_coeffs = (config.sh_degree + 1) ** 2  # e.g., 16 for degree 3
        
        # Keep feature_dim as is for LiDAR compatibility
        # It will be reshaped to [K-1, 3] for camera rendering
        computed_feature_dim = (self._num_sh_coeffs - 1) * 3  # e.g., 45 for degree 3
        
        if config.use_gut_rasterization:
            # Override feature_dim to match SH requirements
            original_feature_dim = config.feature_dim
            config.feature_dim = computed_feature_dim
            
            print(f"[SplatGUT] Camera rendering with 3DGUT:")
            print(f"  - SH degree: {config.sh_degree}")
            print(f"  - SH coefficients: {self._num_sh_coeffs} (DC + {self._num_sh_coeffs - 1} higher-order)")
            print(f"  - feature_dim: {computed_feature_dim} (was {original_feature_dim})")
            print(f"  - Storage format: [N, {computed_feature_dim}] flat")
            print(f"  - Camera format: [N, {self._num_sh_coeffs}, 3] structured")
        
        super().__init__(config, *args, seed_points=seed_points, **kwargs)

    def create_gauss_param_dict(
        self,
        dyn_seed_points_list: List[torch.Tensor],
        static_seed_points_list: List[torch.Tensor],
        flip_actors_at_init: bool = True,
    ) -> Dict[str, nn.Parameter]:
        """Create Gaussian parameters with semantically correct SH initialization.
        
        Key: Initialize in [N, K, 3] format, then flatten to [N, K*3] for storage.
        This ensures SH coefficients have correct semantic structure even in flat format.
        """
        param_dicts = []
        self.xys_grad_norm = None
        self.max_2Dsize = None
        
        feature_dim = self.config.feature_dim  # 45 for sh_degree=3
        
        for i, seed_points in enumerate(dyn_seed_points_list + static_seed_points_list):
            assert seed_points is not None
            assert seed_points.shape[1] == 6, f"Expected seed_points shape [N, 6], got {seed_points.shape}"
            
            flip = False
            if flip_actors_at_init and i < len(dyn_seed_points_list):
                flip = True

            means = nn.Parameter(seed_points[:, :3])
            num_points = means.shape[0]

            # Compute scales from nearest neighbors
            if num_points < 4:
                warnings.warn(f"Actor {i} has less than 4 points, using default scale")
                distances = torch.ones((num_points, 3))
            else:
                distances, _ = self.k_nearest_sklearn(means.data, 3)
                distances = torch.from_numpy(distances)
            
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = nn.Parameter(torch.log(avg_dist.repeat(1, 3) * self.config.init_scale))

            # Random quaternions
            quats = nn.Parameter(random_quat_tensor(num_points))

            # DC component from RGB [N, 3]
            features_dc = nn.Parameter(seed_points[:, 3:] / 255.0)
            
            # ✅ CRITICAL FIX: Initialize with correct SH semantic structure
            # Create as [N, K-1, 3] first, then flatten to [N, (K-1)*3]
            if self.config.use_gut_rasterization:
                num_sh_rest = self._num_sh_coeffs - 1  # 15 for degree 3
                
                # Initialize in structured format [N, K-1, 3]
                features_rest_3d = torch.zeros(
                    num_points,
                    num_sh_rest,
                    3,
                    dtype=seed_points.dtype,
                    device=seed_points.device,
                )
                
                # Flatten to [N, (K-1)*3] for storage (compatible with LiDAR)
                features_rest = nn.Parameter(features_rest_3d.reshape(num_points, -1))
                
                # Verify shape
                assert features_rest.shape == (num_points, feature_dim), \
                    f"features_rest shape mismatch: {features_rest.shape} != ({num_points}, {feature_dim})"
            else:
                # Standard SplatAD initialization (random features)
                features_rest = nn.Parameter(
                    torch.randn(
                        num_points,
                        feature_dim,
                        dtype=seed_points.dtype,
                        device=seed_points.device,
                    )
                )

            opacities = nn.Parameter(
                torch.logit(self.config.init_opacities * torch.ones(num_points, 1))
            )
            
            ids = nn.Parameter(
                torch.full((num_points, 1), min(float(i), len(dyn_seed_points_list))),
                requires_grad=False
            )
            
            # Handle actor flipping
            if flip:
                mirrored_means = means.clone()
                mirrored_means[:, 0] *= -1
                mirrored_quats = quats.clone()
                mirrored_quats[:, 1] *= -1
                
                means = nn.Parameter(torch.cat([means, mirrored_means], dim=0))
                scales = nn.Parameter(torch.cat([scales, scales.clone()], dim=0))
                quats = nn.Parameter(torch.cat([quats, mirrored_quats], dim=0))
                features_dc = nn.Parameter(torch.cat([features_dc, features_dc.clone()], dim=0))
                features_rest = nn.Parameter(torch.cat([features_rest, features_rest.clone()], dim=0))
                opacities = nn.Parameter(torch.cat([opacities, opacities.clone()], dim=0))
                ids = nn.Parameter(torch.cat([ids, ids.clone()], dim=0))

            param_dicts.append({
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
                "id": ids,
            })
        
        # Concatenate all actors/static objects
        result = nn.ParameterDict({
            key: torch.cat([param_dict[key] for param_dict in param_dicts], dim=0)
            for key in param_dicts[0].keys()
        })
        
        if self.config.use_gut_rasterization:
            print(f"[SplatGUT] Created Gaussians:")
            print(f"  - Total points: {result['means'].shape[0]}")
            print(f"  - features_dc shape: {result['features_dc'].shape}")
            print(f"  - features_rest shape: {result['features_rest'].shape} (will reshape to [N, {self._num_sh_coeffs-1}, 3] for camera)")
        
        return result

    def get_camera_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render camera with 3DGUT rasterization using native SH support."""
        if not isinstance(camera, Cameras):
            return {}

        if self.training or self.config.use_camopt_in_eval:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(1 / camera_scale_fac)
        
        K = camera.get_intrinsics_matrices()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(camera_scale_fac)

        render_mode = (
            "RGB"
            if self.config.with_eval3d
            else ("RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB")
        )

        viewmat = get_viewmat(optimized_camera_to_world)
        
        # Handle camera timing and rolling shutter
        camera_times = camera.times
        if camera.metadata is not None and self.config.compensate_rs_camera:
            time_to_center_pixel = camera.metadata.get(
                "time_to_center_pixel",
                torch.zeros((1, 1), device=self.device),
            )
            time_to_center_pixel = (
                time_to_center_pixel + self.camera_velocity_optimizer.get_time_to_center_pixel_adjustment(camera)
            )
            camera_times = camera.times + time_to_center_pixel
        
        means, _ = self._get_actor_adjusted_means(
            self.means, camera_times, self.id, calc_vels=False
        )

        if self.config.use_gut_rasterization:
            # ✅ CRITICAL: Reshape flat features to structured SH format for 3DGUT
            N = self.features_dc.shape[0]
            
            # features_rest: [N, (K-1)*3] -> [N, K-1, 3]
            features_rest_3d = self.features_rest.view(N, self._num_sh_coeffs - 1, 3)
            
            # Concatenate DC and rest: [N, 1, 3] + [N, K-1, 3] = [N, K, 3]
            colors_sh = torch.cat([
                self.features_dc[:, None, :],  # [N, 3] -> [N, 1, 3]
                features_rest_3d               # [N, K-1, 3]
            ], dim=1)
            
            # Verify shape (only in debug mode)
            if torch.is_grad_enabled():  # Only check during training
                expected_shape = (N, self._num_sh_coeffs, 3)
                assert colors_sh.shape == expected_shape, \
                    f"SH shape error: got {colors_sh.shape}, expected {expected_shape}"

            background = self._get_background_color()
            raster_kwargs = self._build_distortion_kwargs(camera)
            
            # Call 3DGUT rasterization with proper SH format
            render, alpha, self.info = gut_rasterization(
                means=means,
                quats=self.quats,
                scales=torch.exp(self.scales),
                opacities=torch.sigmoid(self.opacities).squeeze(-1),
                colors=colors_sh,  # [N, K, 3] - proper SH format
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                near_plane=0.5,
                far_plane=1e10,
                radius_clip=self.config.radius_clip_pix,
                eps2d=0.3,
                sh_degree=self.config.sh_degree,
                packed=False,
                tile_size=16,
                backgrounds=None,
                render_mode=render_mode,
                sparse_grad=False,
                absgrad=self.config.use_absgrad,
                rasterize_mode=self.config.rasterize_mode,
                channel_chunk=128,
                distributed=False,
                camera_model=self.config.camera_model,
                segmented=False,
                covars=None,
                with_ut=self.config.with_ut,
                with_eval3d=self.config.with_eval3d,
                radial_coeffs=raster_kwargs.get("radial_coeffs"),
                tangential_coeffs=raster_kwargs.get("tangential_coeffs"),
                thin_prism_coeffs=raster_kwargs.get("thin_prism_coeffs"),
                rolling_shutter=RollingShutterType.GLOBAL,
                viewmats_rs=None,
            )

            if self.training:
                self.strategy.step_pre_backward(
                    self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                )

            # 3DGUT returns RGB directly (no decoder needed)
            rgb = render[..., :3]
            rgb = rgb + (1 - alpha) * background.view(1, 1, 1, 3)
            rgb = torch.clamp(rgb, 0.0, 1.0)

            depth_im: Optional[torch.Tensor]
            if render_mode == "RGB+ED":
                depth_im = render[..., -1:]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
            else:
                depth_im = None

            if background.shape[0] == 3 and not self.training:
                background = background.expand(H, W, 3)

            return {
                "rgb": rgb.squeeze(0),
                "depth": depth_im.squeeze(0) if depth_im is not None else None,
                "accumulation": alpha.squeeze(0),
                "background": background,
            }
        else:
            # Fall back to parent class implementation (standard SplatAD with decoder)
            return super().get_camera_outputs(camera)

    def _build_distortion_kwargs(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """Extract distortion coefficients from camera for 3DGUT."""
        if camera.distortion_params is None:
            return {}
        
        params = camera.distortion_params
        if params is None:
            return {}
        
        params = params.to(self.device)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        
        if not torch.any(params != 0):
            return {}

        radial = torch.zeros(
            params.shape[:-1] + (6,),
            device=params.device,
            dtype=params.dtype
        )
        radial[..., : min(4, params.shape[-1])] = params[..., : min(4, params.shape[-1])]
        
        tangential = params[..., 4:6] if params.shape[-1] >= 6 else None

        kwargs: Dict[str, torch.Tensor] = {}
        if torch.any(radial != 0):
            kwargs["radial_coeffs"] = radial
        if tangential is not None and torch.any(tangential != 0):
            kwargs["tangential_coeffs"] = tangential
        
        return kwargs

    # LiDAR rendering is completely inherited from SplatAD - no changes needed!
    # The flat [N, feature_dim] format is used directly for lidar_rasterization.
    # This ensures zero impact on LiDAR training.