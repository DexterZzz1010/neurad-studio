"""
3DGUT Adapter: Bridge between PyTorch tensors and 3DGUT C++/CUDA kernels.

This adapter implements the Unscented Transform projection and ray tracing
interfaces, converting SplatAD's Gaussian parameters to 3DGUT's expected format.

Key responsibilities:
    1. Data format conversion (PyTorch Tensor ↔ C++ pointer)
    2. Parameter reorganization (SplatAD format → 3DGUT format)
    3. Rolling shutter state handling
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import warnings


class GUTProjectorAdapter:
    """
    Adapter for 3DGUT Unscented Transform projection.
    
    Replaces EWA splatting with sigma-point based projection that handles:
    - Arbitrary camera distortion models
    - Rolling shutter effects (per-sigma-point extrinsics)
    - Derivative-free projection (no Jacobian required)
    
    Args:
        alpha: UT spread parameter, controls sigma point deviation (default: 1.0)
        beta: UT weight parameter for covariance (default: 0.0)
        kappa: UT secondary parameter (default: 0.0)
        enable_fallback: Use EWA projection if 3DGUT unavailable (default: True)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        kappa: float = 0.0,
        enable_fallback: bool = True,
    ):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.enable_fallback = enable_fallback
        
        # Try importing 3DGUT C++ bindings
        self._init_gut_backend()
    
    def _init_gut_backend(self):
        """Initialize 3DGUT backend or set fallback mode."""
        try:
            # NOTE: Replace with actual 3DGUT Python binding import
            # import threedgut
            # self.gut = threedgut
            # self.available = True
            
            # Temporary: simulate unavailable for testing
            raise ImportError("3DGUT Python bindings not found")
            
        except ImportError:
            self.gut = None
            self.available = False
            
            if self.enable_fallback:
                warnings.warn(
                    "3DGUT not available, falling back to EWA projection. "
                    "Install 3DGUT for UT-based projection support.",
                    RuntimeWarning
                )
            else:
                raise RuntimeError(
                    "3DGUT required but not found. "
                    "Set enable_fallback=True to use EWA projection."
                )
    
    def project(
        self,
        means: Tensor,            # [N, 3]
        quats: Tensor,            # [N, 4]
        scales: Tensor,           # [N, 3]
        opacities: Tensor,        # [N]
        camera_matrix: Tensor,    # [4, 4]
        intrinsics: Tensor,       # [3, 3]
        resolution: Tuple[int, int],
        rolling_shutter_state: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Project 3D Gaussians to 2D using Unscented Transform.
        
        Args:
            means: Gaussian centers in world space
            quats: Quaternion rotations (need not be normalized)
            scales: Scale factors along principal axes
            opacities: Opacity values in [0, 1]
            camera_matrix: Camera-to-world transform [4x4]
            intrinsics: Camera intrinsic matrix [3x3]
            resolution: Image (width, height)
            rolling_shutter_state: Optional dict with keys:
                - 'lin_vel': Linear velocity [3]
                - 'ang_vel': Angular velocity [3]
                - 'time': Rolling shutter duration (scalar)
        
        Returns:
            Dictionary containing:
                - 'means2d': Projected 2D centers [N, 2]
                - 'conics': Inverse 2D covariance, upper triangle [N, 3]
                - 'depths': Z-depths in camera space [N]
                - 'radii': Bounding box radii [N, 2]
                - 'opacities': Potentially compensated opacities [N]
        """
        
        if not self.available:
            # Fallback to EWA projection
            return self._project_ewa_fallback(
                means, quats, scales, opacities,
                camera_matrix, intrinsics, resolution,
                rolling_shutter_state
            )
        
        # Pack data for C++ interface
        particles = self._pack_particles(means, quats, scales, opacities)
        sensor_model = self._make_sensor_model(intrinsics, resolution)
        sensor_state = self._make_sensor_state(camera_matrix, rolling_shutter_state)
        sensor_world_pos = self._extract_camera_position(camera_matrix)
        sensor_matrix = self._to_sensor_matrix(camera_matrix)
        
        # Call 3DGUT UT projection kernel
        projected = self.gut.project_unscented(
            particles=particles,
            sensor_model=sensor_model,
            sensor_world_position=sensor_world_pos,
            sensor_matrix=sensor_matrix,
            sensor_state=sensor_state,
            resolution=resolution,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
        
        # Unpack results to PyTorch tensors
        return self._unpack_projection(projected, means.device)
    
    def _project_ewa_fallback(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        camera_matrix: Tensor,
        intrinsics: Tensor,
        resolution: Tuple[int, int],
        rolling_shutter_state: Optional[Dict],
    ) -> Dict[str, Tensor]:
        """Fallback to gsplat EWA projection when 3DGUT unavailable."""
        from gsplat.cuda._wrapper import fully_fused_projection
        
        W, H = resolution
        viewmat = torch.inverse(camera_matrix)[None]  # [1, 4, 4]
        
        # Extract rolling shutter parameters
        lin_vel = None
        ang_vel = None
        rs_time = None
        if rolling_shutter_state is not None:
            lin_vel = rolling_shutter_state.get('lin_vel')
            ang_vel = rolling_shutter_state.get('ang_vel')
            rs_time = rolling_shutter_state.get('time')
        
        # Call gsplat projection
        radii, means2d, depths, conics, compensations, _ = fully_fused_projection(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmats=viewmat,
            Ks=intrinsics[None],
            width=W,
            height=H,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            rolling_shutter_time=rs_time,
            packed=False,
        )
        
        # Convert to unified format (squeeze batch dimension)
        return {
            'means2d': means2d[0],         # [N, 2]
            'conics': conics[0],           # [N, 3]
            'depths': depths[0],           # [N]
            'radii': radii[0],             # [N, 2]
            'opacities': opacities,        # [N]
        }
    
    # ==================== Data Conversion Helpers ====================
    
    def _pack_particles(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
    ):
        """Convert PyTorch tensors to 3DGUT Particles structure."""
        N = means.shape[0]
        
        # Ensure contiguous memory for C++ interop
        means_c = means.contiguous()
        quats_c = quats.contiguous()
        scales_c = scales.contiguous()
        opacities_c = opacities.contiguous()
        
        # Create 3DGUT Particles struct
        particles = self.gut.Particles()
        particles.count = N
        particles.position_ptr = means_c.data_ptr()
        particles.rotation_ptr = quats_c.data_ptr()
        particles.scale_ptr = scales_c.data_ptr()
        particles.opacity_ptr = opacities_c.data_ptr()
        
        return particles
    
    def _make_sensor_model(self, K: Tensor, resolution: Tuple[int, int]):
        """Convert camera intrinsics to 3DGUT TSensorModel."""
        W, H = resolution
        
        model = self.gut.TSensorModel()
        model.fx = K[0, 0].item()
        model.fy = K[1, 1].item()
        model.cx = K[0, 2].item()
        model.cy = K[1, 2].item()
        model.width = W
        model.height = H
        
        return model
    
    def _make_sensor_state(
        self,
        c2w: Tensor,
        rs_state: Optional[Dict],
    ):
        """Convert rolling shutter state to 3DGUT TSensorState."""
        state = self.gut.TSensorState()
        
        # Rolling shutter parameters
        if rs_state is not None and rs_state.get('time', 0) > 0:
            state.linearVelocity = self._to_vec3(rs_state['lin_vel'])
            state.angularVelocity = self._to_vec3(rs_state['ang_vel'])
            state.rollingShutterTime = rs_state['time'].item()
        else:
            state.linearVelocity = self.gut.vec3(0.0, 0.0, 0.0)
            state.angularVelocity = self.gut.vec3(0.0, 0.0, 0.0)
            state.rollingShutterTime = 0.0
        
        # Camera pose (simplified: no interpolation)
        state.startPose = self._to_sensor_pose(c2w)
        state.endPose = state.startPose
        
        return state
    
    def _extract_camera_position(self, c2w: Tensor):
        """Extract camera position from camera-to-world matrix."""
        return self._to_vec3(c2w[:3, 3])
    
    def _to_sensor_matrix(self, c2w: Tensor):
        """Convert camera-to-world to sensor matrix (world-to-camera)."""
        w2c = torch.inverse(c2w)
        
        # 3DGUT uses [3x4] matrices
        mat = self.gut.mat4x3()
        for i in range(3):
            for j in range(4):
                mat[i][j] = w2c[i, j].item()
        
        return mat
    
    def _to_vec3(self, tensor: Tensor):
        """Convert PyTorch tensor to 3DGUT vec3."""
        if tensor is None:
            return self.gut.vec3(0.0, 0.0, 0.0)
        
        if tensor.dim() == 0:
            v = tensor.item()
            return self.gut.vec3(v, v, v)
        
        return self.gut.vec3(
            tensor[0].item(),
            tensor[1].item(),
            tensor[2].item(),
        )
    
    def _to_sensor_pose(self, c2w: Tensor):
        """Convert 4x4 matrix to 3DGUT TSensorPose."""
        w2c = torch.inverse(c2w)
        
        pose = self.gut.TSensorPose()
        
        # Rotation (3x3)
        for i in range(3):
            for j in range(3):
                pose.rotation[i][j] = w2c[i, j].item()
        
        # Translation (3,)
        pose.translation = self._to_vec3(w2c[:3, 3])
        
        return pose
    
    def _unpack_projection(self, projected, device: torch.device):
        """Convert 3DGUT projection results to PyTorch tensors."""
        N = projected.count
        
        # Allocate output tensors
        means2d = torch.zeros(N, 2, device=device)
        conics = torch.zeros(N, 3, device=device)
        depths = torch.zeros(N, device=device)
        radii = torch.zeros(N, 2, device=device)
        opacities = torch.zeros(N, device=device)
        
        # Copy from C++ to PyTorch
        self.gut.copy_to_tensor(projected.means2d_ptr, means2d)
        self.gut.copy_to_tensor(projected.conics_ptr, conics)
        self.gut.copy_to_tensor(projected.depths_ptr, depths)
        self.gut.copy_to_tensor(projected.radii_ptr, radii)
        self.gut.copy_to_tensor(projected.opacities_ptr, opacities)
        
        return {
            'means2d': means2d,
            'conics': conics,
            'depths': depths,
            'radii': radii,
            'opacities': opacities,
        }


class GUTRayTracer:
    """
    Adapter for 3DGUT Ray Tracing renderer.
    
    Enables secondary ray effects (reflection, refraction) by tracing
    through the Gaussian particle field instead of rasterization.
    
    Note: Significantly slower than rasterization (~3-4x), use only when
    secondary effects are required.
    """
    
    def __init__(self):
        self._init_tracer()
    
    def _init_tracer(self):
        """Initialize 3DGUT ray tracer backend."""
        try:
            # import threedgut
            # self.tracer = threedgut.TGUTRenderer()
            # self.available = True
            
            raise ImportError("3DGUT ray tracer not available")
            
        except ImportError:
            self.tracer = None
            self.available = False
            warnings.warn(
                "3DGUT ray tracer not available. Ray tracing disabled.",
                RuntimeWarning
            )
    
    def trace(
        self,
        projected_gaussians: Dict[str, Tensor],
        features: Tensor,
        width: int,
        height: int,
    ) -> Dict[str, Tensor]:
        """
        Render scene using ray tracing.
        
        Args:
            projected_gaussians: Output from GUTProjectorAdapter.project()
            features: Per-Gaussian feature vectors [N, D]
            width: Image width
            height: Image height
        
        Returns:
            Dictionary containing:
                - 'features': Rendered feature image [H, W, D]
                - 'depth': Depth map [H, W]
                - 'alpha': Alpha channel [H, W]
        """
        
        if not self.available:
            raise RuntimeError(
                "3DGUT ray tracer not available. "
                "Use rasterization or install 3DGUT."
            )
        
        # Generate camera rays
        rays = self._generate_rays(width, height, projected_gaussians)
        
        # Call 3DGUT ray tracer
        output = self.tracer.renderForward(
            rayOrigins=rays['origins'],
            rayDirections=rays['directions'],
            gaussians=projected_gaussians,
            features=features,
        )
        
        return {
            'features': output.radiance[:, :, :features.shape[-1]],
            'depth': output.distance,
            'alpha': 1.0 - output.transmittance,
        }
    
    def _generate_rays(self, W: int, H: int, proj_params: Dict):
        """Generate camera rays for ray tracing."""
        # TODO: Implement ray generation based on projected parameters
        raise NotImplementedError("Ray generation not yet implemented")