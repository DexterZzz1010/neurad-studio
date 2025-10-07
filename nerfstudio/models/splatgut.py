"""
SplatGUT: SplatAD with 3DGUT Unscented Transform Projection.

Minimal invasive extension of SplatAD that replaces EWA splatting
with Unscented Transform for better handling of:
    - Distorted camera models (fisheye, equirectangular)
    - Rolling shutter effects (per-sigma-point extrinsics)
    - Derivative-free projection (no Jacobian required)

Inherits all scene management, optimization, and decoding logic from SplatAD.
Only overrides camera rendering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type

import torch
from torch import Tensor
from typing_extensions import Literal

from nerfstudio.adapters.threedgut_adapter import GUTProjectorAdapter, GUTRayTracer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    """
    Configuration for SplatGUT model.
    
    Inherits all SplatAD parameters (warmup_length, refine_every, etc.)
    and adds 3DGUT-specific options.
    """
    
    _target: Type = field(default_factory=lambda: SplatGUTModel)
    
    # ==================== 3DGUT Projection ====================
    
    use_gut_projection: bool = True
    """Use Unscented Transform projection instead of EWA splatting."""
    
    gut_alpha: float = 1.0
    """UT spread parameter (controls sigma point deviation)."""
    
    gut_beta: float = 0.0
    """UT weight parameter for covariance calculation."""
    
    gut_kappa: float = 0.0
    """UT secondary parameter."""
    
    # ==================== Rendering Mode ====================
    
    use_ray_tracing: bool = False
    """Use ray tracing instead of rasterization (slower, enables secondary rays)."""
    
    enable_fallback: bool = True
    """Fallback to EWA projection if 3DGUT unavailable."""


class SplatGUTModel(SplatADModel):
    """
    SplatAD model with 3DGUT Unscented Transform projection.
    
    Design philosophy (following Linus "good taste"):
        1. Eliminate special cases: Unified interface handles both EWA and UT
        2. Practical: Graceful fallback when 3DGUT unavailable
        3. Simple: Only override what's necessary, inherit everything else
    
    Inheritance hierarchy:
        Model (base_model.py)
        ↓
        ADModel (ad_model.py) - Adds dynamic actors, camera optimization
        ↓
        SplatADModel (splatad.py) - Adds Gaussians, decoders, optimization
        ↓
        SplatGUTModel (THIS CLASS) - Replaces projection method only
    """
    
    config: SplatGUTModelConfig
    
    def populate_modules(self):
        """
        Initialize all modules.
        
        Calls parent to initialize:
            - Gaussian parameters (means, quats, scales, opacities, features)
            - Dynamic actors and scene graph
            - RGB and LiDAR decoders
            - Camera optimizer
        
        Then adds 3DGUT components.
        """
        super().populate_modules()
        self._init_gut_components()
    
    def _init_gut_components(self):
        """Initialize 3DGUT-specific components."""
        self.gut_projector = GUTProjectorAdapter(
            alpha=self.config.gut_alpha,
            beta=self.config.gut_beta,
            kappa=self.config.gut_kappa,
            enable_fallback=self.config.enable_fallback,
        )
        
        if self.config.use_ray_tracing:
            self.gut_ray_tracer = GUTRayTracer()
        else:
            self.gut_ray_tracer = None
    
    # ==================== Camera Rendering (Override) ====================
    
    def get_outputs_for_camera(self, camera: Cameras) -> Dict[str, Tensor]:
        """
        Render camera view using 3DGUT or fallback to EWA.
        
        This is the ONLY method we override from SplatADModel.
        All other functionality (LiDAR, loss, metrics, optimization) inherited.
        
        Args:
            camera: Camera object with intrinsics and extrinsics
        
        Returns:
            Dictionary containing:
                - 'rgb': Rendered RGB image [H, W, 3]
                - 'depth': Depth map [H, W]
                - Additional outputs (accumulation, etc.)
        """
        
        if not isinstance(camera, Cameras):
            raise TypeError(f"Expected Cameras, got {type(camera)}")
        
        # ===== Stage 1: Prepare Camera Parameters (reuse SplatAD logic) =====
        camera_to_world = self._prepare_camera_pose(camera)
        K, W, H = self._prepare_camera_intrinsics(camera)
        colors = self._prepare_gaussian_colors()
        rs_state = self._prepare_rolling_shutter_state(camera)
        
        # ===== Stage 2: Project Gaussians (KEY REPLACEMENT!) =====
        if self.config.use_gut_projection:
            projected = self._gut_projection(camera_to_world, K, W, H, rs_state)
        else:
            projected = self._ewa_projection(camera_to_world, K, W, H, rs_state)
        
        # ===== Stage 3: Render (optional ray tracing) =====
        if self.config.use_ray_tracing and self.gut_ray_tracer is not None:
            render_out = self._ray_trace_render(projected, colors, W, H)
        else:
            render_out = self._rasterize_render(projected, colors, W, H)
        
        # ===== Stage 4: Decode Features (reuse SplatAD decoder) =====
        rgb = self._decode_rgb(render_out['features'], camera, W, H)
        
        # Restore camera resolution if downscaled
        self._restore_camera_resolution(camera)
        
        return {
            'rgb': rgb,
            'depth': render_out['depth'],
            'accumulation': render_out.get('alpha', None),
        }
    
    # ==================== Preparation Helpers (reuse SplatAD) ====================
    
    def _prepare_camera_pose(self, camera: Cameras) -> Tensor:
        """Get optimized camera pose. Reuses SplatAD's camera optimization."""
        if self.training or self.config.use_camopt_in_eval:
            assert camera.shape[0] == 1, "Only one camera at a time"
            return self.camera_optimizer.apply_to_camera(camera)
        else:
            return camera.camera_to_worlds
    
    def _prepare_camera_intrinsics(
        self,
        camera: Cameras
    ) -> Tuple[Tensor, int, int]:
        """Prepare camera intrinsics and resolution."""
        camera_scale_fac = self._get_downscale_factor()
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(1 / camera_scale_fac)
        
        K = camera.get_intrinsics_matrices()
        W = int(camera.width.item())
        H = int(camera.height.item())
        
        self.last_size = (H, W)  # Store for later use
        
        return K, W, H
    
    def _prepare_gaussian_colors(self) -> Tensor:
        """Concatenate SH coefficients into color features."""
        return torch.cat([self.features_dc, self.features_rest], dim=-1)
    
    def _prepare_rolling_shutter_state(self, camera: Cameras) -> Optional[Dict]:
        """Extract rolling shutter parameters from camera."""
        camera_times = camera.times
        
        if camera_times is None:
            return None
        
        # Get velocity from optimizer
        if self.training:
            lin_vel, ang_vel = self.camera_velocity_optimizer.apply_to_camera_velocity(camera)
        else:
            if self.config.use_camopt_in_eval:
                lin_vel, ang_vel = self.camera_velocity_optimizer.apply_to_camera_velocity(camera)
            else:
                lin_vel = None
                ang_vel = None
        
        # Compute rolling shutter time
        if lin_vel is not None and ang_vel is not None:
            rs_time = camera_times.max() - camera_times.min()
            return {
                'lin_vel': lin_vel,
                'ang_vel': ang_vel,
                'time': rs_time,
            }
        
        return None
    
    def _restore_camera_resolution(self, camera: Cameras):
        """Restore camera resolution if it was downscaled."""
        camera_scale_fac = self._get_downscale_factor()
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(camera_scale_fac)
    
    # ==================== Projection Methods ====================
    
    def _gut_projection(
        self,
        c2w: Tensor,
        K: Tensor,
        W: int,
        H: int,
        rs_state: Optional[Dict],
    ) -> Dict[str, Tensor]:
        """3DGUT Unscented Transform projection."""
        return self.gut_projector.project(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            camera_matrix=c2w[0] if c2w.dim() == 3 else c2w,
            intrinsics=K[0] if K.dim() == 3 else K,
            resolution=(W, H),
            rolling_shutter_state=rs_state,
        )
    
    def _ewa_projection(
        self,
        c2w: Tensor,
        K: Tensor,
        W: int,
        H: int,
        rs_state: Optional[Dict],
    ) -> Dict[str, Tensor]:
        """Fallback to original EWA splatting projection."""
        from gsplat.cuda._wrapper import fully_fused_projection
        
        viewmat = torch.inverse(c2w)[None]
        
        # Extract rolling shutter parameters
        lin_vel = rs_state['lin_vel'] if rs_state else None
        ang_vel = rs_state['ang_vel'] if rs_state else None
        rs_time = rs_state['time'] if rs_state else None
        
        radii, means2d, depths, conics, compensations, _ = fully_fused_projection(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            viewmats=viewmat,
            Ks=K[None],
            width=W,
            height=H,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            rolling_shutter_time=rs_time,
            packed=False,
        )
        
        return {
            'means2d': means2d[0],
            'conics': conics[0],
            'depths': depths[0],
            'radii': radii[0],
            'opacities': self.opacities,
        }
    
    # ==================== Rendering Methods ====================
    
    def _rasterize_render(
        self,
        projected: Dict[str, Tensor],
        colors: Tensor,
        W: int,
        H: int,
    ) -> Dict[str, Tensor]:
        """Rasterization rendering (fast, primary rays only)."""
        from gsplat.rendering import rasterization
        
        # Reshape for rasterization API
        means2d = projected['means2d'][None]
        conics = projected['conics'][None]
        opacities = projected['opacities'][None]
        colors_batch = colors[None]
        
        # Dummy viewmat and K (already projected)
        viewmat = torch.eye(4, device=colors.device)[None]
        K = torch.eye(3, device=colors.device)[None]
        
        render_colors, render_alphas, meta = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=opacities[0],
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            render_mode="RGB+ED",
        )
        
        return {
            'features': render_colors[0],
            'depth': meta['render_depth'][0] if 'render_depth' in meta else projected['depths'],
            'alpha': render_alphas[0],
        }
    
    def _ray_trace_render(
        self,
        projected: Dict[str, Tensor],
        colors: Tensor,
        W: int,
        H: int,
    ) -> Dict[str, Tensor]:
        """Ray tracing rendering (slow, supports secondary rays)."""
        if self.gut_ray_tracer is None:
            raise RuntimeError("Ray tracer not initialized")
        
        return self.gut_ray_tracer.trace(projected, colors, W, H)
    
    # ==================== Decoding ====================
    
    def _decode_rgb(
        self,
        features: Tensor,
        camera: Cameras,
        W: int,
        H: int,
    ) -> Tensor:
        """Decode features to RGB using SplatAD's CNN decoder."""
        # Compute view directions
        from nerfstudio.models.splatad import get_ray_dirs_pinhole
        
        c2w = self._prepare_camera_pose(camera)
        ray_dirs = get_ray_dirs_pinhole(camera, W, H, c2w)
        
        # Apply CNN decoder
        rgb = self.rgb_decoder(features, ray_dirs)
        
        return rgb.squeeze(0)  # Remove batch dimension
    
    # ==================== Inherited Methods (DO NOT OVERRIDE) ====================
    
    # The following methods are inherited from SplatADModel and work unchanged:
    # - get_lidar_outputs()      # LiDAR rendering unchanged
    # - get_metrics_dict()       # Metrics computation unchanged
    # - get_loss