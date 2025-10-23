"""
SplatGUT model built on top of SplatAD with 3DGUT rasterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatad import RGB2SH, SplatADModel, SplatADModelConfig
from nerfstudio.models.splatfacto import get_viewmat

try:
    from gsplat_original.cuda._wrapper import RollingShutterType
    from gsplat_original.rendering import rasterization as gut_rasterization
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Please install the gsplat_original package with 3DGUT support.") from exc


@dataclass
class SplatGUTModelConfig(SplatADModelConfig):
    """Configuration for the SplatGUT model."""

    _target: Type = field(default_factory=lambda: SplatGUTModel)

    with_ut: bool = True
    """Enable Unscented Transform projection."""

    with_eval3d: bool = True
    """Evaluate splats in 3D (slower but more accurate)."""

    camera_model: Literal["pinhole", "fisheye", "ortho", "ftheta"] = "pinhole"
    """Camera model passed to the 3DGUT rasterizer."""

    sh_degree: int = 3
    """Spherical harmonic degree used for image rendering."""

    lidar_feature_dim: int = 16
    """Feature dimension available for lidar decoding."""

    def __post_init__(self):
        # Reserve feature_rest capacity exclusively for lidar features.
        self.feature_dim = self.lidar_feature_dim


class SplatGUTModel(SplatADModel):
    """SplatAD variant that renders images via 3DGUT rasterization."""

    config: SplatGUTModelConfig

    def __init__(
        self,
        *args,
        seed_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        super().__init__(*args, seed_points=seed_points, **kwargs)
        self._num_sh_coeffs = (self.config.sh_degree + 1) ** 2

    def create_gauss_param_dict(
        self,
        dyn_seed_points_list: List[torch.Tensor],
        static_seed_points_list: List[torch.Tensor],
        flip_actors_at_init: bool = True,
    ):
        """Reshape Gaussian features to store SH coefficients and lidar descriptors separately."""
        param_dict = super().create_gauss_param_dict(
            dyn_seed_points_list, static_seed_points_list, flip_actors_at_init=flip_actors_at_init
        )

        features_dc_param = param_dict["features_dc"]
        num_gauss = features_dc_param.shape[0]
        device = features_dc_param.device
        dtype = features_dc_param.dtype

        sh_coeffs = torch.zeros((num_gauss, self._num_sh_coeffs, 3), device=device, dtype=dtype)
        sh_coeffs[:, 0, :] = RGB2SH(features_dc_param.data.clamp(0.0, 1.0))
        param_dict["features_dc"] = torch.nn.Parameter(sh_coeffs.view(num_gauss, -1))

        features_rest_param = param_dict["features_rest"]
        if features_rest_param.shape[1] != self.config.lidar_feature_dim:
            lidar_feats = torch.randn(
                (num_gauss, self.config.lidar_feature_dim), device=device, dtype=dtype
            )
            param_dict["features_rest"] = torch.nn.Parameter(lidar_feats)

        return param_dict

    def get_camera_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render RGB images with 3DGUT rasterization."""
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
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        render_mode = (
            "RGB"
            if self.config.with_eval3d
            else ("RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB")
        )

        viewmat = get_viewmat(optimized_camera_to_world)
        camera_times = camera.times
        means, _ = self._get_actor_adjusted_means(self.means, camera_times, self.id, calc_vels=False)

        colors_arg = self.features_dc.view(-1, self._num_sh_coeffs, 3)
        sh_degree = self.config.sh_degree

        background = self._get_background_color()
        raster_kwargs = self._build_distortion_kwargs(camera)

        render, alpha, self.info = gut_rasterization(
            means=means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities).squeeze(-1),
            colors=colors_arg,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            near_plane=0.5,
            far_plane=1e10,
            radius_clip=self.config.radius_clip_pix,
            eps2d=0.3,
            sh_degree=sh_degree,
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
            ftheta_coeffs=None,
            rolling_shutter=RollingShutterType.GLOBAL,
            viewmats_rs=None,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        rgb = render[..., :3]
        if render_mode == "RGB+ED":
            depth_im = render[..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
        else:
            depth_im = None
        rgb = rgb + (1 - alpha) * background.view(1, 1, 1, 3)
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        out: Dict[str, Union[torch.Tensor, List]] = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im.squeeze(0) if depth_im is not None else None,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,
        }
        return out

    def _build_distortion_kwargs(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """Extract distortion coefficients compatible with 3DGUT if available."""
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

        radial = torch.zeros(params.shape[:-1] + (6,), device=params.device, dtype=params.dtype)
        radial[..., : min(4, params.shape[-1])] = params[..., : min(4, params.shape[-1])]
        tangential = params[..., 4:6] if params.shape[-1] >= 6 else None

        kwargs: Dict[str, torch.Tensor] = {}
        if torch.any(radial != 0):
            kwargs["radial_coeffs"] = radial
        if tangential is not None and torch.any(tangential != 0):
            kwargs["tangential_coeffs"] = tangential
        return kwargs
