"""
3DGUT Adapter for SplatAD

Single responsibility: Convert SplatAD data to 3DGUT format and delegate to Tracer.
No rendering logic, no parameter conversion - those already exist in threedgut_tracer.

Design:
    - CamerasToBatchConverter: Cameras → Batch 
    - GaussiansWrapper: SplatAD params → 3DGUT Gaussians interface 
    - Call threedgut_tracer.Tracer.render() directly 

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
        
        3DGUT's Tracer.__create_camera_parameters() will handle the rest.
        
        Args:
            camera_model: "pinhole", "fisheye", or "auto"
            fisheye_distortion: [k1, k2, k3, k4] or None (auto-detect from camera)
        """
        H, W = rays_o.shape[:2]
        c2w = camera_to_world[0] if camera_to_world.dim() == 3 else camera_to_world
        
        # Extract intrinsics
        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        
        # Infer camera model if "auto"
        if camera_model == "auto":
            # Check if camera has distortion_params attribute
            dist_params = getattr(camera, 'distortion_params', None)
            if dist_params is not None and not (dist_params == 0).all():
                camera_model = "fisheye"
                fisheye_distortion = dist_params.cpu().numpy() if isinstance(dist_params, Tensor) else dist_params
            else:
                camera_model = "pinhole"
        
        # Build 3DGUT camera model parameters
        if camera_model == "fisheye":
            # Get distortion coefficients
            if fisheye_distortion is None:
                # Try to read from camera object
                dist_params = getattr(camera, 'distortion_params', None)
                if dist_params is None:
                    raise ValueError(
                        "Fisheye model requires distortion_params. "
                        "Either set in dataparser or in config.fisheye_distortion"
                    )
                fisheye_distortion = dist_params.cpu().numpy() if isinstance(dist_params, Tensor) else dist_params
            
            # Ensure 4 coefficients [k1, k2, k3, k4]
            if len(fisheye_distortion) < 4:
                fisheye_distortion = np.pad(fisheye_distortion, (0, 4 - len(fisheye_distortion)))
            
            params = OpenCVFisheyeCameraModelParameters(
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([fx, fy], dtype=np.float32),
                radial_coeffs=fisheye_distortion[:4].astype(np.float32),
                resolution=np.array([W, H], dtype=np.int64),
                max_angle=CamerasToBatchConverter._compute_fisheye_max_angle(fx, fy, cx, cy, W, H),
                shutter_type=ShutterType.GLOBAL,
            )
            intrinsics_key = "intrinsics_OpenCVFisheyeCameraModelParameters"
        else:
            # Pinhole model
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
        gaussians.num_gaussians, .positions, .n_active_features
        .get_rotation(), .get_scale(), .get_density(), .get_features()
        .background()
    
    We just forward SplatAD's self.means, self.quats, etc.
    """
    
    def __init__(self, model):
        self.model = model
        self.positions = model.means
        
        # Number of Gaussians
        self.num_gaussians = model.means.shape[0]
        
        # Calculate n_active_features from actual feature tensors
        # SplatAD structure:
        #   features_dc:   [N, 3]      - DC component (1 SH coeff per channel)
        #   features_rest: [N, K, 3]   - Higher order SH (K additional coeffs per channel)
        # Total: (1 + K) * 3 features per Gaussian
        
        if hasattr(model, 'features_rest') and model.features_rest.dim() > 1:
            # [N, K, 3] -> K additional SH coefficients
            n_sh_coeffs = 1 + model.features_rest.shape[1]  # DC + rest
            self.n_active_features = n_sh_coeffs * 3
        else:
            # Only DC component
            self.n_active_features = model.features_dc.shape[-1]  # Usually 3
    
    def get_rotation(self):
        return self.model.quats
    
    def get_scale(self):
        return self.model.scales
    
    def get_density(self):
        return self.model.opacities
    
    def get_features(self):
        """
        Concat SH coefficients to match 3DGUT format.
        
        SplatAD format:
            features_dc:   [N, 3]      - DC component
            features_rest: [N, K, 3]   - Higher order SH
        
        3DGUT expects: [N, (1+K)*3]
        """
        if self.model.features_rest.dim() > 1 and self.model.features_rest.shape[1] > 0:
            # Reshape [N, K, 3] -> [N, K*3]
            N = self.model.features_rest.shape[0]
            features_rest_flat = self.model.features_rest.reshape(N, -1)
            
            # Concatenate [N, 3] + [N, K*3] = [N, (1+K)*3]
            return torch.cat([
                self.model.features_dc,
                features_rest_flat
            ], dim=1)
        else:
            # Only DC component
            return self.model.features_dc
    
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
            2. Build Batch (with camera model from config)
            3. Call Tracer.render()
            4. Done
        """
        # Wrap Gaussians (zero-copy)
        gaussians = GaussiansWrapper(model)
        
        # Convert camera to Batch (read camera_model from model.config)
        fisheye_dist = None
        if hasattr(model.config, 'fisheye_distortion') and model.config.fisheye_distortion is not None:
            fisheye_dist = np.array(model.config.fisheye_distortion, dtype=np.float32)
        
        batch = CamerasToBatchConverter.convert(
            camera=camera,
            camera_to_world=c2w,
            rays_o=rays_o,
            rays_d=rays_d,
            camera_model=model.config.camera_model,  # From config!
            fisheye_distortion=fisheye_dist,
        )
        
        # Delegate to 3DGUT (THE ONLY RENDERING CALL)
        outputs = self.tracer.render(gaussians, batch, train=model.training)
        
        return {
            'rgb': outputs['pred_rgb'].squeeze(0),
            'depth': outputs['pred_dist'].squeeze(0),
            'alpha': outputs['pred_opacity'].squeeze(0),
        }