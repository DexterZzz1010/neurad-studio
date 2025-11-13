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
"""COLMAP dataparser that feeds COLMAP reconstructions into the AD pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, CameraType, Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.data.dataparsers.ad_dataparser import (
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE

from threedgrut.datasets.utils import (
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
    qvec_to_so3,
)


def _resolve_dataset_path(root: Path, subpath: str) -> Path:
    """Resolve dataset-relative paths while allowing absolute overrides."""
    path = Path(subpath)
    return path if path.is_absolute() else (root / path)


def _colmap_camera_params(camera) -> Tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from a COLMAP camera record."""
    model = camera.model.upper()
    params = camera.params
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "FOV"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")
    return fx, fy, cx, cy


@dataclass
class ColmapDataParserConfig(ADDataParserConfig):
    """Configuration for the COLMAP dataparser."""

    _target: Type = field(default_factory=lambda: ColmapDataParser)

    cameras: Tuple[str, ...] = ("colmap",)
    lidars: Tuple[str, ...] = tuple()
    load_cuboids: bool = False
    annotation_interval: float = 1.0

    colmap_model_path: str = "sparse/0"
    """Relative path to the COLMAP model directory (contains cameras.[bin|txt], images.[bin|txt])."""
    use_binary_model: bool = True
    """Whether to read binary (.bin) or text (.txt) COLMAP files."""
    images_path: str = "images"
    """Relative path from dataset root to the folder that stores RGB images."""
    allowed_camera_ids: Tuple[int, ...] = tuple()
    """Optional subset of COLMAP camera IDs to load. Empty tuple loads every camera."""
    image_name_filter: Optional[str] = None
    """Optional fnmatch-style pattern for filtering image filenames."""
    ignore_missing_images: bool = True
    """Skip frames whose image file is missing instead of raising."""
    downscale_factor: float = 1.0
    """Scale factor applied to COLMAP intrinsics if images were downscaled after reconstruction."""
    synthetic_time_interval: float = 1.0
    """Spacing between consecutive timestamps when COLMAP does not provide real capture times."""
    apply_opencv_to_nerfstudio: bool = True
    """Whether to flip COLMAP's (right-down-forward) axes into NeuRAD's (right-forward-up)."""
    camera_id_to_name: Dict[int, str] = field(default_factory=dict)
    """Optional mapping from COLMAP camera ID to a human readable sensor name."""
    scene_box_padding: float = 10.0
    """Meters of padding to add around camera centers when deriving the scene bounding box."""


@dataclass
class ColmapDataParser(ADDataParser):
    """Dataparser that exposes static COLMAP reconstructions to the AD pipeline."""

    config: ColmapDataParserConfig

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        dataset_root = Path(self.config.data).expanduser()
        model_root = _resolve_dataset_path(dataset_root, self.config.colmap_model_path)
        images_root = _resolve_dataset_path(dataset_root, self.config.images_path)
        intrinsics_path = model_root / ("cameras.bin" if self.config.use_binary_model else "cameras.txt")
        extrinsics_path = model_root / ("images.bin" if self.config.use_binary_model else "images.txt")
        if not intrinsics_path.exists() or not extrinsics_path.exists():
            raise FileNotFoundError(f"Missing COLMAP files under {model_root}")

        if self.config.use_binary_model:
            cam_intrinsics = read_colmap_intrinsics_binary(str(intrinsics_path))
            cam_extrinsics = read_colmap_extrinsics_binary(str(extrinsics_path))
        else:
            cam_intrinsics = read_colmap_intrinsics_text(str(intrinsics_path))
            cam_extrinsics = read_colmap_extrinsics_text(str(extrinsics_path))

        camera_id_filter = set(self.config.allowed_camera_ids)
        fx_list: List[float] = []
        fy_list: List[float] = []
        cx_list: List[float] = []
        cy_list: List[float] = []
        widths: List[int] = []
        heights: List[int] = []
        camera_types: List[int] = []
        poses: List[Tensor] = []
        image_filenames: List[Path] = []
        sensor_idxs: List[int] = []
        camera_ids: List[int] = []

        camera_id_to_sensor_idx: Dict[int, int] = {}
        sensor_names: List[str] = []

        def _sensor_idx(camera_id: int) -> int:
            if camera_id not in camera_id_to_sensor_idx:
                sensor_idx = len(camera_id_to_sensor_idx)
                camera_id_to_sensor_idx[camera_id] = sensor_idx
                sensor_names.append(self.config.camera_id_to_name.get(camera_id, f"camera_{camera_id}"))
            return camera_id_to_sensor_idx[camera_id]

        for image in cam_extrinsics:
            if camera_id_filter and image.camera_id not in camera_id_filter:
                continue
            if self.config.image_name_filter and not fnmatch(image.name, self.config.image_name_filter):
                continue
            intrinsic = cam_intrinsics.get(image.camera_id)
            if intrinsic is None:
                CONSOLE.print(f"[yellow]Missing intrinsics for camera id {image.camera_id}; skipping {image.name}")
                continue
            img_path = self._resolve_image_file(dataset_root, images_root, Path(image.name))
            if not img_path.exists():
                message = f"[yellow]Image file {img_path} not found."
                if self.config.ignore_missing_images:
                    CONSOLE.print(f"{message} Skipping frame.")
                    continue
                raise FileNotFoundError(message)

            fx, fy, cx, cy = _colmap_camera_params(intrinsic)
            scale = float(self.config.downscale_factor)
            if scale <= 0:
                raise ValueError("downscale_factor must be > 0")
            fx /= scale
            fy /= scale
            cx /= scale
            cy /= scale
            width = int(round(intrinsic.width / scale))
            height = int(round(intrinsic.height / scale))

            pose = self._colmap_pose_to_c2w(image)

            fx_list.append(float(fx))
            fy_list.append(float(fy))
            cx_list.append(float(cx))
            cy_list.append(float(cy))
            widths.append(width)
            heights.append(height)
            camera_types.append(CAMERA_MODEL_TO_TYPE.get(intrinsic.model, CameraType.PERSPECTIVE).value)
            poses.append(torch.from_numpy(pose[:3, :4]).float())
            image_filenames.append(img_path)
            camera_ids.append(image.camera_id)
            sensor_idxs.append(_sensor_idx(image.camera_id))

        if not poses:
            raise RuntimeError("No COLMAP frames were loaded. Check filters and paths.")

        self.config.cameras = tuple(sensor_names)

        times = torch.arange(len(poses), dtype=torch.float64).unsqueeze(-1)
        times *= float(self.config.synthetic_time_interval)

        cameras = Cameras(
            camera_to_worlds=torch.stack(poses, dim=0),
            fx=torch.tensor(fx_list, dtype=torch.float32),
            fy=torch.tensor(fy_list, dtype=torch.float32),
            cx=torch.tensor(cx_list, dtype=torch.float32),
            cy=torch.tensor(cy_list, dtype=torch.float32),
            width=torch.tensor(widths, dtype=torch.int32),
            height=torch.tensor(heights, dtype=torch.int32),
            camera_type=torch.tensor(camera_types, dtype=torch.int64),
            times=times,
            metadata={
                "sensor_idxs": torch.tensor(sensor_idxs, dtype=torch.int32).unsqueeze(-1),
                "camera_ids": torch.tensor(camera_ids, dtype=torch.int32).unsqueeze(-1),
            },
        )
        return cameras, image_filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        lidars = Lidars(
            lidar_to_worlds=torch.zeros((0, 3, 4), dtype=torch.float32),
            times=torch.zeros((0, 1), dtype=torch.float64),
            metadata={"sensor_idxs": torch.zeros((0, 1), dtype=torch.int32)},
        )
        return lidars, []

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        return []

    def _get_actor_trajectories(self):
        return []

    def _get_radars(self):
        return []

    def _colmap_pose_to_c2w(self, image) -> np.ndarray:
        rot = image.qvec_to_so3() if hasattr(image, "qvec_to_so3") else qvec_to_so3(image.qvec)
        if not isinstance(rot, np.ndarray) or rot.shape != (3, 3):
            rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        tvec = np.asarray(image.tvec, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rot
        w2c[:3, 3] = tvec
        c2w = np.linalg.inv(w2c)
        if self.config.apply_opencv_to_nerfstudio:
            c2w[:3, :3] = c2w[:3, :3] @ OPENCV_TO_NERFSTUDIO
        return c2w

    def _resolve_image_file(self, dataset_root: Path, images_root: Path, rel_name: Path) -> Path:
        """Resolve an image filename that may already encode part of the directory structure."""
        if rel_name.is_absolute():
            return rel_name
        candidates = [
            dataset_root / rel_name,
            images_root / rel_name,
            rel_name,
        ]
        seen = set()
        for candidate in candidates:
            key = candidate.resolve().as_posix() if candidate.exists() else candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
        # Fall back to relative to images root if nothing matched yet
        return images_root / rel_name

    def _compute_scene_box(self, cameras: Cameras, lidars: Lidars) -> SceneBox:
        if len(lidars):
            return super()._compute_scene_box(cameras, lidars)
        if not len(cameras):
            raise RuntimeError("Cannot derive a scene box without any cameras.")
        positions = cameras.camera_to_worlds[:, :3, 3]
        padding = float(self.config.scene_box_padding)
        aabb_min = positions.min(dim=0)[0] - padding
        aabb_max = positions.max(dim=0)[0] + padding
        aabb_min[2] = float(self.config.scene_box_height[0])
        aabb_max[2] = float(self.config.scene_box_height[1])
        aabb = torch.stack([aabb_min, aabb_max], dim=0)
        return SceneBox(aabb=aabb)
