# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Render fisheye images from a pinhole checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import tyro
import tyro.extras
import yaml
import tempfile

from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.scripts.render import (
    _render_trajectory_video,
    streamline_ad_config,
)
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


def _parse_opencv_fisheye_model(model_line: str) -> Tuple[int, int, float, float, float, float, Tuple[float, float, float, float]]:
    """Parse COLMAP style OPENCV_FISHEYE intrinsics."""
    tokens = model_line.replace(",", " ").split()
    if len(tokens) != 11:
        raise ValueError(f"Expected 11 tokens in camera model description, got {len(tokens)}: {model_line}")
    model = tokens[0].upper()
    if model != "OPENCV_FISHEYE":
        raise ValueError(f"Expected OPENCV_FISHEYE model, got {model}")
    width, height = int(tokens[1]), int(tokens[2])
    fx, fy, cx, cy = map(float, tokens[3:7])
    k1, k2, k3, k4 = map(float, tokens[7:11])
    return width, height, fx, fy, cx, cy, (k1, k2, k3, k4)


def _build_fisheye_cameras(
    base_cameras: Cameras,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: Sequence[float],
) -> Cameras:
    """Copy poses from base cameras but override intrinsics and camera type."""
    device = base_cameras.camera_to_worlds.device
    dtype = base_cameras.camera_to_worlds.dtype
    num_cams = len(base_cameras)

    def _full_like(template: torch.Tensor, value: float) -> torch.Tensor:
        return template.new_full(template.shape, value)

    fx_tensor = _full_like(base_cameras.fx, fx)
    fy_tensor = _full_like(base_cameras.fy, fy)
    cx_tensor = _full_like(base_cameras.cx, cx)
    cy_tensor = _full_like(base_cameras.cy, cy)
    width_tensor = base_cameras.width.new_full(base_cameras.width.shape, width)
    height_tensor = base_cameras.height.new_full(base_cameras.height.shape, height)

    distortion_tensor = torch.zeros((num_cams, 6), dtype=dtype, device=device)
    dist_values = torch.tensor(distortion, dtype=dtype, device=device)
    distortion_tensor[:, :4] = dist_values

    camera_type_tensor = base_cameras.camera_type.new_full(
        base_cameras.camera_type.shape, CameraType.FISHEYE.value
    )

    return Cameras(
        camera_to_worlds=base_cameras.camera_to_worlds,
        fx=fx_tensor,
        fy=fy_tensor,
        cx=cx_tensor,
        cy=cy_tensor,
        width=width_tensor,
        height=height_tensor,
        distortion_params=distortion_tensor,
        camera_type=camera_type_tensor,
        times=base_cameras.times,
        metadata=base_cameras.metadata,
    )


def _fix_config_load_dir(config_path: Path) -> Path:
    """
    Fix load_dir in config if it's a string instead of Path object.
    Returns path to fixed config (or original if no fix needed).
    """
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    
    # Check if load_dir exists and is a string
    # Config might be an object, not a dict
    if hasattr(config, 'load_dir'):
        load_dir = getattr(config, 'load_dir')
        if isinstance(load_dir, str):
            CONSOLE.print(f"[yellow]Fixing load_dir: converting string '{load_dir}' to Path object[/yellow]")
            setattr(config, 'load_dir', Path(load_dir))
            
            # Write to temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.yml', prefix='config_fixed_')
            import os
            os.close(temp_fd)  # Close the file descriptor
            
            with open(temp_path, 'w') as f:
                yaml.dump(config, f)
            
            CONSOLE.print(f"[green]Fixed config written to {temp_path}[/green]")
            return Path(temp_path)
        elif load_dir is None:
            CONSOLE.print(f"[red]ERROR: load_dir is None in config[/red]")
            raise ValueError("load_dir is None in config. Cannot proceed without checkpoint directory.")
    
    return config_path


@dataclass
class ColormapOptions:
    """Colormap options."""
    colormap: Literal["default", "turbo", "viridis", "magma", "inferno", "cividis", "gray", "pca"] = "default"
    """The colormap to use."""
    normalize: bool = False
    """Whether to normalize the input tensor image."""
    colormap_min: float = 0
    """Minimum value for the output colormap."""
    colormap_max: float = 1
    """Maximum value for the output colormap."""
    invert: bool = False
    """Whether to invert the output colormap."""


@dataclass
class RenderFisheyeDataset:
    """Render dataset poses with an OPENCV_FISHEYE camera model."""

    load_config: Path
    """Path to config YAML file."""

    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""

    output_format: Literal["images", "video"] = "images"
    """Output format (images or video)."""

    image_format: Literal["jpeg", "png"] = "png"
    """Image format."""

    jpeg_quality: int = 100
    """JPEG quality."""

    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""

    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis."""

    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""

    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""

    colormap_options: ColormapOptions = field(default_factory=ColormapOptions)
    """Colormap options."""

    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""

    check_occlusions: bool = False
    """If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other."""

    fisheye_model: str = (
        "OPENCV_FISHEYE 1936 1216 396.65756 396.65756 "
        "969.8259 616.7724 0.07276304811239243 -0.002853460144251585 "
        "0.006127878092229366 -0.0020023139659315348"
    )
    """Camera model string exported from COLMAP (model width height fx fy cx cy k1 k2 k3 k4)."""

    pose_source: Literal["train", "eval"] = "eval"
    """Use train or eval split to fetch poses."""

    image_indices: Optional[List[int]] = field(default=None)
    """Subset of frame indices to render."""

    fps: float = 24.0
    """Target output framerate when generating videos."""

    def main(self) -> None:
        if self.output_format == "video":
            install_checks.check_ffmpeg_installed()

        # Fix config if load_dir is a string
        fixed_config_path = _fix_config_load_dir(self.load_config)

        _, pipeline, _, _ = eval_setup(
            fixed_config_path,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
        )

        dataset = (
            pipeline.datamanager.train_dataset if self.pose_source == "train" else pipeline.datamanager.eval_dataset
        )
        if dataset is None:
            raise RuntimeError(f"{self.pose_source} dataset is not available in this checkpoint.")

        cameras = dataset.cameras
        if self.image_indices:
            cameras = cameras[self.image_indices]
        if len(cameras) == 0:
            raise RuntimeError("No cameras selected for rendering.")

        width, height, fx, fy, cx, cy, distortion = _parse_opencv_fisheye_model(self.fisheye_model)
        fisheye_cameras = _build_fisheye_cameras(cameras, width, height, fx, fy, cx, cy, distortion)

        seconds = max(len(fisheye_cameras) / max(self.fps, 1e-3), 1e-3)

        _render_trajectory_video(
            pipeline,
            fisheye_cameras,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=None,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )

        CONSOLE.print(
            ":sparkles: Finished fisheye rendering using "
            f"{self.pose_source} split with {len(fisheye_cameras)} frames."
        )


def entrypoint() -> None:
    """CLI entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderFisheyeDataset).main()


if __name__ == "__main__":
    entrypoint()