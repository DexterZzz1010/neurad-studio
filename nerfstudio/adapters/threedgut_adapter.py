"""
3DGUT Adapter for SplatAD - The Right Way.

Single responsibility: Convert SplatAD data to 3DGUT format and delegate to Tracer.
No rendering logic, no parameter conversion - those already exist in threedgut_tracer.

Design:
    - CamerasToBatchConverter: Cameras → Batch (30 lines)
    - GaussiansWrapper: SplatAD params → 3DGUT Gaussians interface (40 lines)
    - Call threedgut_tracer.Tracer.render() directly (3 lines)

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
        camera_type: str = "pinhole",
        fisheye_coeffs: Optional[np.ndarray] = None,
    ) -> Batch:
        """
        Build Batch matching 3DGUT's __getitem__ output.
        
        3DGUT's Tracer.__create_camera_parameters() will handle the rest.
        """
        H, W = rays_o.shape[:2]
        c2w = camera_to_world[0] if camera_to_world.dim() == 3 else camera_to_world
        
        # Extract intrinsics
        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        
        # Build intrinsics dict (format expected by Tracer.__create_camera_parameters)
        if camera_type == "fisheye":
            params = OpenCVFisheyeCameraModelParameters(
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fx, fy], dtype=np.float32),
                radial_coeffs=fisheye_coeffs.astype(np.float32),
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
            'rays_ori': rays_o.unsqueeze(0),
            'rays_dir': rays_d.unsqueeze(0),
            'T_to_world': c2w.unsqueeze(0),
            intrinsics_key: params.to_dict(),
        })
    
    @staticmethod
    def _compute_fisheye_max_angle(fx, fy, cx, cy, W, H):
        """From dataset_colmap.py - compute FOV for fisheye."""
        max_radius = np.sqrt(max(
            cx**2 + cy**2,
            (W - cx)**2 + cy**2,
            cx**2 + (H - cy)**2,
            (W - cx)**2 + (H - cy)**2,
        ))
        return float(max(2.0 * max_radius / fx, 2.0 * max_radius / fy) / 2.0)


class GaussiansWrapper:
    """
    Wrap SplatAD parameters in 3DGUT's Gaussians interface.
    
    3DGUT expects:
        gaussians.positions, .get_rotation(), .get_scale(), 
        .get_density(), .get_features(), .n_active_features
    
    We just forward SplatAD's self.means, self.quats, etc.
    """
    
    def __init__(self, model):
        self.model = model
        self.positions = model.means
        self.n_active_features = model.num_sh_degree * 3  # SH coefficients
    
    def get_rotation(self):
        return self.model.quats
    
    def get_scale(self):
        return self.model.scales
    
    def get_density(self):
        return self.model.opacities
    
    def get_features(self):
        """Concat SH coefficients."""
        return torch.cat([
            self.model.features_dc,
            self.model.features_rest
        ], dim=-1)
    
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
        from omegaconf import OmegaConf
        self.tracer = threedgut_tracer.Tracer(OmegaConf.create(config))
    
    def render(self, model, camera, rays_o, rays_d, c2w) -> Dict[str, Tensor]:
        """
        Render using 3DGUT.
        
        Steps:
            1. Wrap Gaussians
            2. Build Batch
            3. Call Tracer.render()
            4. Done
        """
        # Wrap Gaussians (zero-copy)
        gaussians = GaussiansWrapper(model)
        
        # Convert camera to Batch
        batch = CamerasToBatchConverter.convert(
            camera=camera,
            camera_to_world=c2w,
            rays_o=rays_o,
            rays_d=rays_d,
            camera_type=getattr(camera, 'camera_type', 'pinhole'),
            fisheye_coeffs=getattr(camera, 'distortion_params', None),
        )
        
        # Delegate to 3DGUT (THE ONLY RENDERING CALL)
        outputs = self.tracer.render(gaussians, batch, train=model.training)
        
        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }