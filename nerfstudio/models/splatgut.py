"""
SplatGUT: Hybrid rendering.

Design:
    - Camera: 3DGUT ray tracing
    - LiDAR: gsplat (inherited)
"""

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig
from nerfstudio.cameras.cameras import Cameras


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    _target: Type = field(default_factory=lambda: SplatGUTModel)

    sh_degree: int = 2
    target_radius: float = 200.0
    use_ray_tracing: bool = True
    k_buffer_size: int = 32
    ut_alpha: float = 1.0
    ut_beta: float = 0.0
    camera_model: str = "pinhole"
    fisheye_distortion: list = None
    use_camopt_in_eval: bool = True


class SplatGUTModel(SplatADModel):
    config: SplatGUTModelConfig

    def populate_modules(self):
        """Initialize modules after parent creates gauss_params."""
        super().populate_modules()
        
        # Setup SH from gauss_params
        self._setup_sh_from_gauss_params()
        
        # Initialize 3DGUT renderer
        if self.config.use_ray_tracing:
            self._init_gut_renderer()

    def _setup_sh_from_gauss_params(self):
        """
        Create SH parameters from gauss_params dictionary.
        
        SplatAD stores Gaussians in self.gauss_params dict, not self.gaussians.
        """
        L = self.config.sh_degree
        M = (L + 1) ** 2
        
        # Find means to determine N and device
        means = None
        features_dc = None
        features_rest_key = None
        
        for key, val in self.gauss_params.items():
            if 'means' in key.lower():
                means = val
            elif 'features_dc' in key.lower():
                features_dc = val
            elif 'features_rest' in key.lower():
                features_rest_key = key
        
        if means is None:
            raise RuntimeError("Cannot find 'means' in gauss_params")
        
        N = means.shape[0]
        device = means.device
        dtype = means.dtype
        
        # Create high-order SH parameter
        self.sh_high_order = nn.Parameter(
            torch.zeros(N, 3, M - 1, device=device, dtype=dtype)
        )
        
        # Replace features_rest in gauss_params
        if features_rest_key is not None:
            self.gauss_params[features_rest_key] = self.sh_high_order
        
        print(f"[SplatGUT] SH initialized: N={N}, L={L}, M={M}")

    def _init_gut_renderer(self):
        """Initialize 3DGUT renderer."""
        try:
            from nerfstudio.adapters.threedgut_adapter import GUT3DRenderer, THREEDGUT_AVAILABLE
        except ImportError as e:
            raise ImportError(f"3DGUT adapter not found: {e}")
        
        if not THREEDGUT_AVAILABLE:
            raise ImportError("3DGUT not installed")
        
        L = self.config.sh_degree
        config = {
            'render': {
                'method': '3dgut',
                'particle_radiance_sph_degree': L,
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
        
        self.gut_renderer = GUT3DRenderer(config)
        print(f"[SplatGUT] 3DGUT renderer ready (L={L}, k={self.config.k_buffer_size})")

    def get_param_groups(self) -> Dict[str, list]:
        """
        Define optimizer parameter groups.
        
        SplatAD uses gauss_params dict, not self.gaussians.
        """
        groups = super().get_param_groups()
        
        # Add SH high-order to sh group
        sh_group = groups.get("sh", [])
        
        # Add features_dc from gauss_params
        for key, val in self.gauss_params.items():
            if 'features_dc' in key.lower() and val not in sh_group:
                sh_group.append(val)
        
        # Add sh_high_order
        if hasattr(self, 'sh_high_order') and self.sh_high_order not in sh_group:
            sh_group.append(self.sh_high_order)
        
        groups["sh"] = sh_group
        
        return groups

    def get_outputs_for_camera(self, camera: Cameras) -> Dict[str, Tensor]:
        """Camera rendering: 3DGUT or gsplat."""
        if not self.config.use_ray_tracing:
            return super().get_outputs_for_camera(camera)
        
        return self._render_camera_with_gut(camera)

    def _render_camera_with_gut(self, camera: Cameras) -> Dict[str, Tensor]:
        """Render camera with 3DGUT."""
        # Camera optimization
        if self.config.use_camopt_in_eval or self.training:
            camera = self.camera_optimizer.apply_to_camera(camera)
        
        # Downscaling
        scale = self._get_downscale_factor()
        if scale != 1:
            camera.rescale_output_resolution(1 / scale)
        
        W, H = int(camera.width.item()), int(camera.height.item())
        c2w = camera.camera_to_worlds
        
        # Generate rays
        rays_o, rays_d = self._generate_rays(camera, W, H, c2w)
        
        # Render with 3DGUT
        outputs = self.gut_renderer.render(
            model=self,
            camera=camera,
            rays_o=rays_o,
            rays_d=rays_d,
            c2w=c2w,
            target_radius=self.config.target_radius,
            gut_sh_degree=self.config.sh_degree,
        )
        
        # Composite
        rgb, alpha, depth = outputs['rgb'], outputs['alpha'], outputs['depth']
        
        alpha = alpha.unsqueeze(-1) if alpha.ndim == 2 else alpha
        depth = depth.squeeze(-1) if depth.ndim == 3 else depth
        
        bg = self._get_background_color()
        if bg.ndim == 1:
            bg = bg.view(1, 1, 3).expand(H, W, 3)
        
        rgb = torch.clamp(rgb * alpha + (1 - alpha) * bg, 0, 1)
        
        if scale != 1:
            camera.rescale_output_resolution(scale)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": alpha.squeeze(-1),
            "background": bg,
        }

    @torch.no_grad()
    def _generate_rays(self, camera: Cameras, W: int, H: int, 
                       c2w: Tensor) -> tuple[Tensor, Tensor]:
        """Generate rays (pinhole or fisheye)."""
        device = c2w.device
        
        fx, fy = float(camera.fx.item()), float(camera.fy.item())
        cx, cy = float(camera.cx.item()), float(camera.cy.item())
        
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        R = c2w[0, :3, :3] if c2w.dim() == 3 else c2w[:3, :3]
        t = c2w[0, :3, 3] if c2w.dim() == 3 else c2w[:3, 3]
        
        if self.config.camera_model == "fisheye":
            dirs_cam = self._fisheye_unproject(xx, yy, fx, fy, cx, cy, camera)
        else:
            dirs_cam = torch.stack([
                (xx - cx) / fx,
                (yy - cy) / fy,
                torch.ones_like(xx)
            ], dim=-1)
        
        rays_d = F.normalize(dirs_cam @ R.T, dim=-1)
        rays_o = t[None, None, :].expand(H, W, 3)
        
        return rays_o.contiguous(), rays_d.contiguous()

    def _fisheye_unproject(self, xx: Tensor, yy: Tensor,
                          fx: float, fy: float, cx: float, cy: float,
                          camera: Cameras) -> Tensor:
        """Fisheye unprojection (Kannala-Brandt)."""
        device = xx.device
        
        if self.config.fisheye_distortion is not None:
            k = torch.tensor(self.config.fisheye_distortion, 
                           device=device, dtype=torch.float32)
        elif hasattr(camera, 'distortion_params') and camera.distortion_params is not None:
            k = camera.distortion_params.to(device).float()
        else:
            raise ValueError("Fisheye requires distortion_params [k1,k2,k3,k4]")
        
        if k.numel() < 4:
            k = torch.cat([k, torch.zeros(4 - k.numel(), device=device)])
        k = k[:4]
        
        u, v = (xx - cx) / fx, (yy - cy) / fy
        rho = torch.sqrt(u*u + v*v)
        phi = torch.atan2(v, u)
        
        theta = rho.clone()
        for _ in range(4):
            t2, t4, t6, t8 = theta**2, theta**4, theta**6, theta**8
            poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
            f = theta * poly - rho
            fp = poly + theta * (2*k[0]*theta + 4*k[1]*t2 + 6*k[2]*t4 + 8*k[3]*t6)
            fp = torch.where(fp.abs() > 1e-6, fp, torch.ones_like(fp) * 1e-6)
            theta = theta - f / fp
        
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)
        
        return torch.stack([
            sin_t * cos_p,
            sin_t * sin_p,
            cos_t
        ], dim=-1)