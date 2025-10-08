"""
SplatGUT: SplatAD + 3DGUT Ray Tracing.

Inherits ALL SplatAD logic (scene graph, optimization, losses).
Only overrides camera rendering: EWA splatting → Unscented Transform ray tracing.

Linus principles:
    1. Eliminate special cases: Same Model interface for both renderers
    2. Practical: Graceful degradation if 3DGUT unavailable
    3. Simple: Override ONE method, ~50 LOC
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from torch import Tensor

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig
from nerfstudio.cameras.cameras import Cameras


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    """SplatGUT config - adds ray tracing + camera model selection."""
    
    _target: Type = field(default_factory=lambda: SplatGUTModel)
    
    use_ray_tracing: bool = True
    """Enable 3DGUT ray tracing. Slower but handles distortion/rolling shutter."""
    
    camera_model: str = "pinhole"
    """Camera model: 'pinhole', 'fisheye', or 'auto' (infer from distortion_params)."""
    
    # Fisheye distortion coefficients (if camera_model="fisheye" and not in dataparser)
    fisheye_distortion: Optional[list] = None
    """[k1, k2, k3, k4] for fisheye. If None, read from camera.distortion_params."""
    
    # 3DGUT-specific rendering parameters
    k_buffer_size: int = 32
    """Max Gaussians per ray (memory vs quality trade-off)."""
    
    ut_alpha: float = 1.0
    """Unscented Transform spread parameter (default: 1.0)."""
    
    ut_beta: float = 0.0
    """Unscented Transform kurtosis parameter (default: 0.0)."""


class SplatGUTModel(SplatADModel):
    """
    SplatAD with 3DGUT.
    
    Inheritance chain:
        Model → ADModel → SplatADModel → SplatGUTModel
    
    Inherited (unchanged):
        - populate_modules(): Gaussians, scene graph, decoders, optimizers
        - get_outputs(): RGB + LiDAR rendering pipeline
        - get_loss_dict(): All loss functions
        - get_metrics_dict(): PSNR, SSIM, etc.
        - Dynamic actors, MCMC refinement, camera optimization
    
    Overridden (this file):
        - get_outputs_for_camera(): EWA → UT ray tracing
    """
    
    config: SplatGUTModelConfig
    
    def populate_modules(self):
        """Initialize all modules (calls parent for everything)."""
        super().populate_modules()
        
        if self.config.use_ray_tracing:
            # Lazy import to avoid circular dependency
            try:
                from nerfstudio.adapters.threedgut_adapter import GUT3DRenderer, THREEDGUT_AVAILABLE
            except ImportError as e:
                print(f"[ERROR] Failed to import threedgut_adapter: {e}")
                print("[WARN] Falling back to rasterization")
                self.config.use_ray_tracing = False
                return
            
            if not THREEDGUT_AVAILABLE:
                print("[WARN] 3DGUT not available, falling back to rasterization")
                self.config.use_ray_tracing = False
                return
            
            # Get sh_degree safely (might be sh_degree_max or num_sh_degree)
            sh_degree = getattr(self.config, 'sh_degree', 
                               getattr(self.config, 'sh_degree_max',
                                      getattr(self, 'num_sh_degree', 3)))
            
            # Build 3DGUT config (minimal required params)
            gut_config = {
                'render': {
                    'method': '3dgut',
                    'particle_radiance_sph_degree': sh_degree,
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
            print(f"[INFO] 3DGUT initialized:")
            print(f"  - Camera model: {self.config.camera_model}")
            print(f"  - SH degree: {sh_degree}")
            print(f"  - K-buffer size: {self.config.k_buffer_size}")
            print(f"  - UT params: α={self.config.ut_alpha}, β={self.config.ut_beta}")
    
    def get_outputs_for_camera(self, camera: Cameras) -> Dict[str, Tensor]:
        """
        Render camera with 3DGUT ray tracing.
        
        Pipeline:
            1. Generate rays (nerfstudio helper)
            2. Apply camera optimization (SplatAD logic)
            3. Render with 3DGUT
            4. Decode RGB (SplatAD decoder)
        """
        
        if not self.config.use_ray_tracing:
            # Fallback to parent's rasterization
            return super().get_outputs_for_camera(camera)
        
        # Step 1: Prepare camera (SplatAD logic)
        if self.training or self.config.use_camopt_in_eval:
            camera = self.camera_optimizer.apply_to_camera(camera)
        
        # Handle downscaling
        scale = self._get_downscale_factor()
        if scale != 1:
            camera.rescale_output_resolution(1 / scale)
        
        W, H = int(camera.width.item()), int(camera.height.item())
        c2w = camera.camera_to_worlds
        
        # Step 2: Generate rays (standard pinhole for now)
        rays_o, rays_d = self._generate_camera_rays(camera, W, H, c2w)
        
        # Step 3: Render with 3DGUT (THE CORE CHANGE)
        outputs = self.gut_renderer.render(
            model=self,  # Pass self so GaussiansWrapper can access params
            camera=camera,
            rays_o=rays_o,
            rays_d=rays_d,
            c2w=c2w,
        )
        
        # Step 4: Decode RGB if decoder present
        if hasattr(self, 'rgb_decoder'):
            from nerfstudio.models.splatad import get_ray_dirs_pinhole
            ray_dirs = get_ray_dirs_pinhole(camera, W, H, c2w)
            rgb = self.rgb_decoder(outputs['rgb'], ray_dirs).squeeze(0)
        else:
            rgb = outputs['rgb']
        
        # Restore resolution
        if scale != 1:
            camera.rescale_output_resolution(scale)
        
        return {
            'rgb': rgb,
            'depth': outputs['depth'],
            'accumulation': outputs['alpha'],
        }
    
    def _generate_camera_rays(self, camera, W, H, c2w):
        """
        Generate camera rays based on camera model.
        
        For pinhole: Standard perspective projection
        For fisheye: Kannala-Brandt undistortion (if available)
        """
        device = c2w.device
        fx = camera.fx.item()
        fy = camera.fy.item()
        cx = camera.cx.item()
        cy = camera.cy.item()
        
        # Generate pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Check if fisheye
        if self.config.camera_model == "fisheye" or (
            self.config.camera_model == "auto" and 
            hasattr(camera, 'distortion_params') and 
            camera.distortion_params is not None
        ):
            # Use fisheye undistortion
            rays_d = self._generate_fisheye_rays(x, y, fx, fy, cx, cy, camera, device)
        else:
            # Use pinhole projection
            dirs_cam = torch.stack([
                (x - cx) / fx,
                (y - cy) / fy,
                torch.ones_like(x)
            ], dim=-1)
            
            # Transform to world space
            R = c2w[0, :3, :3] if c2w.dim() == 3 else c2w[:3, :3]
            rays_d = (dirs_cam @ R.T)
            rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        
        # Origin (same for both)
        t = c2w[0, :3, 3] if c2w.dim() == 3 else c2w[:3, 3]
        rays_o = t[None, None, :].expand(H, W, 3)
        
        return rays_o, rays_d
    
    def _generate_fisheye_rays(self, x, y, fx, fy, cx, cy, camera, device):
        """
        Generate fisheye rays using Kannala-Brandt undistortion.
        
        Matches 3DGUT's kannala_unproject_pixels_to_rays.
        Reference: threedgrut/datasets/utils.py
        """
        # Get distortion coefficients [k1, k2, k3, k4]
        if self.config.fisheye_distortion is not None:
            k = torch.tensor(self.config.fisheye_distortion, device=device, dtype=torch.float32)
        elif hasattr(camera, 'distortion_params'):
            k = camera.distortion_params.to(device).float()
        else:
            raise ValueError("Fisheye requires distortion_params")
        
        if len(k) < 4:
            k = torch.cat([k, torch.zeros(4 - len(k), device=device)])
        k = k[:4]
        
        # Pixel to normalized coordinates
        u_norm = (x - cx) / fx
        v_norm = (y - cy) / fy
        
        # Radial distance in image plane
        rho = torch.sqrt(u_norm**2 + v_norm**2)
        
        # Azimuthal angle
        phi = torch.atan2(v_norm, u_norm)
        
        # Newton-Raphson iteration to solve: theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) = rho
        theta = rho.clone()
        for _ in range(4):
            a = k[0] * theta**2
            b = k[1] * theta**4
            c = k[2] * theta**6
            d = k[3] * theta**8
            
            # Residual: f(theta) = theta*(1+a+b+c+d) - rho
            residual = theta * (1 + a + b + c + d) - rho
            
            # Derivative: f'(theta) = 1 + 3*a + 5*b + 7*c + 9*d
            derivative = 1 + 3*a + 5*b + 7*c + 9*d
            
            # Avoid division by zero
            derivative = torch.where(
                derivative.abs() > 1e-6,
                derivative,
                torch.ones_like(derivative) * 1e-6
            )
            
            # Newton step
            theta = theta - residual / derivative
        
        # Convert spherical (theta, phi) to 3D direction in camera space
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        dirs_cam = torch.stack([
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        ], dim=-1)
        
        # Transform to world space
        c2w = camera.camera_to_worlds
        if self.training or self.config.use_camopt_in_eval:
            c2w = self.camera_optimizer.apply_to_camera(camera).camera_to_worlds
        
        R = c2w[0, :3, :3] if c2w.dim() == 3 else c2w[:3, :3]
        rays_d = (dirs_cam @ R.T)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        
        return rays_d