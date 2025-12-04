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

import bisect
import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Literal

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
    read_colmap_points3D_binary,
    read_colmap_points3D_text,
    sample_points3d_data,
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
    lidars: Tuple[str, ...] = ("colmap_points",)
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
    points3d_path: Optional[str] = None
    """Optional override for the COLMAP points3D file (relative to dataset root)."""
    use_binary_points: Optional[bool] = None
    """When None, follow use_binary_model. Otherwise force binary/text point loading."""
    max_point_cloud_points: int = 2_000_000
    """Maximum number of points kept from the COLMAP cloud."""
    pointcloud_downsample_method: Literal["random", "farthest"] = "random"
    """Strategy used when downsizing the COLMAP point cloud."""
    min_point_cloud_points: int = 1000
    """Generate fallback seeds if fewer valid points remain after sanitization."""
    fallback_pointcloud_size: int = 50000
    """Number of fallback seeds to synthesize when COLMAP data is invalid."""
    fallback_pointcloud_padding: float = 5.0
    """Padding added to the camera bounding box when sampling fallback seeds."""

    camera_timestamps_path: Optional[str] = None
    """Optional path (relative to dataset root) that maps image paths to timestamps."""
    camera_timestamp_reference_sensor: Optional[str] = None
    """Sensor prefix (e.g. 'FISHF') used as the reference timeline for alignment."""
    masks_path: Optional[str] = None
    """Optional path to per-image masks (mirrors images_path layout)."""
    lidar_frames_path: Optional[str] = None
    """Directory that stores raw LiDAR frames (frame_XXXXX.pcd)."""
    lidar_timestamps_path: Optional[str] = None
    """NumPy array storing timestamps for each LiDAR frame."""
    lidar_to_camera_transform_path: Optional[str] = None
    """Optional 4x4 matrix describing the LiDAR-to-reference-camera transform."""
    reference_pose_file: Optional[str] = None
    """Optional path to images_ref_rs.txt that lists reference->camera poses."""
    reference_sensor_name: Optional[str] = None
    """Sensor name in the reference pose file that corresponds to the desired camera (e.g. FISHF)."""
    lidar_quaternion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """LiDAR quaternion given as (w, x, y, z) in the reference coordinate frame."""
    lidar_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """LiDAR translation (meters) in the reference coordinate frame."""
    vehicle_lidar_quaternion: Optional[Tuple[float, float, float, float]] = None
    """LiDAR quaternion (wxyz) in vehicle coordinate frame; if set with vehicle_camera_quaternion/translation, l2c is computed as inv(T_c2v) @ T_l2v."""
    vehicle_lidar_translation: Optional[Tuple[float, float, float]] = None
    vehicle_camera_quaternion: Optional[Tuple[float, float, float, float]] = None
    vehicle_camera_translation: Optional[Tuple[float, float, float]] = None


@dataclass
class ColmapDataParser(ADDataParser):
    """Dataparser that exposes static COLMAP reconstructions to the AD pipeline."""

    config: ColmapDataParserConfig

    _time_key_precision: float = 1e-6

    def _resolve_optional_path(self, dataset_root: Path, path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        path = _resolve_dataset_path(dataset_root, path_str)
        return path if path.exists() else None

    def _matrix_from_quaternion_and_translation(
        self, quaternion: Tuple[float, float, float, float], translation: Tuple[float, float, float]
    ) -> torch.Tensor:
        if len(quaternion) != 4:
            raise ValueError("Quaternion must contain 4 elements (qw, qx, qy, qz).")
        if len(translation) != 3:
            raise ValueError("Translation must contain 3 elements.")
        rot = qvec_to_so3(np.asarray(quaternion, dtype=np.float64))
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = torch.from_numpy(rot.astype(np.float32))
        mat[:3, 3] = torch.tensor(translation, dtype=torch.float32)
        return mat

    def _parse_reference_pose_text_file(self, path: Path) -> Dict[str, torch.Tensor]:
        transforms: Dict[str, torch.Tensor] = {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
        except OSError as exc:
            raise RuntimeError(f"Failed to read reference pose file: {path}") from exc
        if not lines:
            return transforms
        cursor = 0
        try:
            sensor_count = int(lines[cursor].split()[0])
            cursor += 1
        except (ValueError, IndexError):
            sensor_count = 0
        for _ in range(sensor_count):
            if cursor >= len(lines):
                break
            tokens = lines[cursor].split()
            cursor += 1
            if len(tokens) < 10:
                continue
            name = tokens[-1]
            try:
                quaternion = tuple(float(value) for value in tokens[1:5])
                translation = tuple(float(value) for value in tokens[5:8])
            except ValueError:
                continue
            transforms[name] = self._matrix_from_quaternion_and_translation(quaternion, translation)
        return transforms

    def _parse_reference_pose_json(self, path: Path) -> Dict[str, torch.Tensor]:
        transforms: Dict[str, torch.Tensor] = {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to parse reference pose json: {path}") from exc
        if not isinstance(payload, list):
            return transforms
        for rig in payload:
            cameras = rig.get("cameras", [])
            for camera in cameras:
                name = camera.get("image_prefix") or camera.get("name")
                if not name:
                    continue
                name = name.rstrip("/")
                qvec = camera.get("rel_qvec") or camera.get("qvec")
                tvec = camera.get("rel_tvec") or camera.get("tvec")
                if qvec is None or tvec is None or len(qvec) != 4 or len(tvec) != 3:
                    continue
                transforms[name] = self._matrix_from_quaternion_and_translation(
                    tuple(float(v) for v in qvec), tuple(float(v) for v in tvec)
                )
        return transforms

    def _load_reference_sensor_transform(self, dataset_root: Path) -> Optional[torch.Tensor]:
        path = self._resolve_optional_path(dataset_root, self.config.reference_pose_file)
        sensor_name = self.config.reference_sensor_name
        if path is None or not sensor_name:
            return None
        if path.suffix.lower() == ".json":
            transforms = self._parse_reference_pose_json(path)
        else:
            transforms = self._parse_reference_pose_text_file(path)
        sensor_key = sensor_name.rstrip("/")
        return transforms.get(sensor_key)

    def _load_image_timestamp_table(self, dataset_root: Path) -> Dict[str, float]:
        table: Dict[str, float] = {}
        path = self._resolve_optional_path(dataset_root, self.config.camera_timestamps_path)
        if path is None:
            return table
        try:
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                if isinstance(payload, dict):
                    iterable = payload.values()
                else:
                    iterable = payload
                for entry in iterable:
                    rel_path = entry.get("path") or entry.get("image") or entry.get("name")
                    timestamp = entry.get("timestamp")
                    if rel_path is None or timestamp is None:
                        continue
                    normalized = self._normalize_timestamp_value(timestamp)
                    if normalized is None:
                        continue
                    table[str(rel_path)] = normalized
                    table[Path(rel_path).name] = normalized
            else:
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        _, ts, key = parts[0], parts[1], parts[2]
                        normalized = self._normalize_timestamp_value(ts)
                        if normalized is None:
                            continue
                        table[key] = normalized
        except Exception as exc:  # pylint: disable=broad-except
            CONSOLE.log(f"[yellow]Failed to parse camera timestamps from {path}: {exc}")
        return table

    def _normalize_timestamp_value(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        magnitude = abs(numeric)
        if magnitude > 1e15:
            return numeric * 1e-9  # assume nanoseconds
        if magnitude > 1e12:
            return numeric * 1e-6  # assume microseconds
        if magnitude > 1e9:
            return numeric  # already seconds
        if magnitude > 1e6:
            return numeric * 1e-3
        return numeric

    def _finalize_camera_times(
        self,
        raw_timestamps: List[Optional[float]],
        sensor_names: List[str],
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算相机的相对时间,同时保存绝对时间用于LiDAR对齐.
        
        关键点: 所有传感器共享同一个时间原点 (_time_origin)
        """
        valid = [ts for ts in raw_timestamps if ts is not None]
        origin = min(valid) if valid else 0.0
        self._time_origin = origin  # 全局时间原点

        ref_sensor = self.config.camera_timestamp_reference_sensor or (sensor_names[0] if sensor_names else None)
        reference_entries: List[Tuple[float, torch.Tensor]] = []
        relative_times: List[float] = []
        absolute_times: List[float] = []

        for idx, (timestamp, sensor_name) in enumerate(zip(raw_timestamps, sensor_names)):
            value = timestamp
            if value is None:
                value = origin + idx * float(self.config.synthetic_time_interval)
            relative = float(value) - origin
            relative_times.append(relative)
            absolute_times.append(float(value))
            if ref_sensor and sensor_name == ref_sensor:
                reference_entries.append((relative, poses[idx, :3, :4].detach().cpu()))

        if ref_sensor and not reference_entries and sensor_names:
            fallback_sensor = sensor_names[0]
            CONSOLE.log(f"[yellow]Reference sensor '{ref_sensor}' not found, using '{fallback_sensor}' as fallback")
            for idx, sensor_name in enumerate(sensor_names):
                if sensor_name == fallback_sensor:
                    reference_entries.append((relative_times[idx], poses[idx, :3, :4].detach().cpu()))
            ref_sensor = fallback_sensor
            
        reference_entries.sort(key=lambda item: item[0])
        self._reference_sensor_name = ref_sensor
        self._reference_pose_times_array = (
            np.array([item[0] for item in reference_entries], dtype=np.float64) if reference_entries else np.array([])
        )
        self._reference_pose_mats = [item[1] for item in reference_entries]
        self._reference_pose_lookup = {self._time_key(time): pose for time, pose in reference_entries}

        # 保存用于LiDAR对齐
        self._camera_relative_times = relative_times
        self._camera_absolute_times = absolute_times

        return torch.tensor(relative_times, dtype=torch.float64).unsqueeze(-1)

    def _time_key(self, value: float) -> int:
        return int(round(value / self._time_key_precision))

    def _lookup_reference_pose(self, aligned_time: float) -> Optional[torch.Tensor]:
        lookup = getattr(self, "_reference_pose_lookup", None)
        if lookup:
            pose = lookup.get(self._time_key(aligned_time))
            if pose is not None:
                return pose
        ref_times = getattr(self, "_reference_pose_times_array", None)
        ref_poses = getattr(self, "_reference_pose_mats", None)
        if ref_times is None or ref_poses is None or len(ref_poses) == 0:
            return None
        idx = bisect.bisect_left(ref_times, aligned_time)
        candidates: List[Tuple[float, torch.Tensor]] = []
        if idx < len(ref_poses):
            candidates.append((abs(ref_times[idx] - aligned_time), ref_poses[idx]))
        if idx > 0:
            candidates.append((abs(ref_times[idx - 1] - aligned_time), ref_poses[idx - 1]))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _load_lidar_to_camera_transform(self, dataset_root: Path) -> torch.Tensor:
        """
        加载LiDAR到相机的外参,优先级:
        1. vehicle-frame extrinsics (如果4个参数都提供)
        2. lidar_to_camera_transform_path
        3. reference_pose_file + lidar_quaternion/translation
        4. 单位矩阵 (警告)
        """
        # 优先: 从vehicle坐标系计算
        if (
            self.config.vehicle_lidar_quaternion
            and self.config.vehicle_lidar_translation
            and self.config.vehicle_camera_quaternion
            and self.config.vehicle_camera_translation
        ):
            try:
                t_l2v = self._matrix_from_quaternion_and_translation(
                    self.config.vehicle_lidar_quaternion, self.config.vehicle_lidar_translation
                )
                t_c2v = self._matrix_from_quaternion_and_translation(
                    self.config.vehicle_camera_quaternion, self.config.vehicle_camera_translation
                )
                l2c = torch.linalg.inv(t_c2v) @ t_l2v
                CONSOLE.log("[green]Computed LiDAR->Camera from vehicle-frame extrinsics")
                return l2c
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError("Failed to compute LiDAR->Camera from vehicle-frame extrinsics") from exc

        # 次选: 直接加载外参文件
        path = self._resolve_optional_path(dataset_root, self.config.lidar_to_camera_transform_path)
        if path is not None:
            try:
                if path.suffix.lower() == ".npy":
                    matrix = np.load(path)
                else:
                    with open(path, "r", encoding="utf-8") as file:
                        matrix = json.load(file)
                matrix = np.asarray(matrix, dtype=np.float32)
                if matrix.shape != (4, 4):
                    raise ValueError(f"Expected a 4x4 matrix, received shape {matrix.shape}")
                CONSOLE.log(f"[green]Loaded LiDAR->Camera from {path}")
                return torch.from_numpy(matrix)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"Failed to read LiDAR extrinsic from {path}") from exc

        # Fallback: reference_pose_file
        reference_transform = self._load_reference_sensor_transform(dataset_root)
        if reference_transform is not None:
            try:
                lidar_to_reference = self._matrix_from_quaternion_and_translation(
                    self.config.lidar_quaternion, self.config.lidar_translation
                )
            except ValueError as exc:
                raise RuntimeError("Invalid LiDAR quaternion or translation configuration.") from exc
            CONSOLE.log("[yellow]Using reference_pose_file for LiDAR transform")
            return torch.matmul(reference_transform, lidar_to_reference)

        # 最后: 单位矩阵 (应该警告!)
        CONSOLE.log("[yellow]WARNING: No LiDAR extrinsic found, using identity matrix")
        return torch.eye(4, dtype=torch.float32)

    def _load_lidar_timestamps(self, dataset_root: Path) -> List[Optional[float]]:
        path = self._resolve_optional_path(dataset_root, self.config.lidar_timestamps_path)
        if path is None:
            return []
        try:
            raw = np.load(path)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to read LiDAR timestamps from {path}") from exc
        flat = np.asarray(raw).reshape(-1)
        return [self._normalize_timestamp_value(float(value)) for value in flat]

    def _align_lidar_times(self, raw_times: List[Optional[float]]) -> List[float]:
        """
        将LiDAR时间戳对齐到全局时间原点 (_time_origin).
        这保证了相机和LiDAR的相对时间在同一时间轴上.
        """
        origin = getattr(self, "_time_origin", 0.0)
        aligned: List[float] = []
        for idx, raw_time in enumerate(raw_times):
            value = raw_time
            if value is None:
                value = origin + idx * float(self.config.synthetic_time_interval)
            aligned.append(float(value) - origin)
        return aligned

    def _fill_missing_lidar_times(self, raw_times: List[Optional[float]], start_value: float) -> List[float]:
        """用合成间隔填充None时间戳,保持绝对时间"""
        times: List[float] = []
        last = None
        for idx, raw_time in enumerate(raw_times):
            if raw_time is None:
                if last is None:
                    value = start_value + idx * float(self.config.synthetic_time_interval)
                else:
                    value = last + float(self.config.synthetic_time_interval)
            else:
                value = float(raw_time)
            times.append(float(value))
            last = float(value)
        return times

    def _parse_pcd_xyz(self, path: Path) -> np.ndarray:
        with open(path, "rb") as file:
            header: Dict[str, str] = {}
            while True:
                line = file.readline()
                if not line:
                    raise ValueError(f"Malformed PCD file: {path}")
                text = line.decode("utf-8").strip()
                if not text or text.startswith("#"):
                    continue
                key, *rest = text.split(" ", 1)
                if key.upper() == "DATA":
                    data_type = rest[0].strip().lower() if rest else ""
                    break
                header[key.lower()] = rest[0].strip() if rest else ""
            if data_type != "binary":
                raise ValueError(f"Unsupported PCD DATA type '{data_type}' in {path}")
            fields = header.get("fields", "").split()
            sizes = [int(val) for val in header.get("size", "").split()]
            types = header.get("type", "").split()
            counts = [int(val) for val in header.get("count", "").split()]
            points = int(header.get("points", header.get("width", "0")))
            if not fields or len(fields) != len(sizes):
                raise ValueError(f"Invalid PCD header in {path}")
            if points <= 0:
                raise ValueError(f"PCD file '{path}' has no points.")
            if any(c != 1 for c in counts):
                raise ValueError(f"Multi-count PCD fields are not supported (file: {path}).")
            if any(t.upper() != "F" or s != 4 for t, s in zip(types, sizes)):
                raise ValueError(f"Only 32-bit float PCD files are supported (file: {path}).")
            raw = np.fromfile(file, dtype=np.float32, count=points * len(fields))
            if raw.size != points * len(fields):
                raise ValueError(f"Unexpected data size while reading {path}")
            raw = raw.reshape(points, len(fields))
            field_to_index = {name: idx for idx, name in enumerate(fields)}
            try:
                xyz = np.stack(
                    [raw[:, field_to_index["x"]], raw[:, field_to_index["y"]], raw[:, field_to_index["z"]]],
                    axis=1,
                )
            except KeyError as exc:
                raise KeyError(f"Missing required XYZ fields in {path}") from exc
        return xyz

    def _load_dynamic_lidars(self, dataset_root: Path) -> Tuple[Lidars, List[Path]]:
        """
        加载动态LiDAR扫描,通过时间戳与相机对齐.
        
        关键逻辑:
        1. 用绝对时间进行匹配 (避免原点偏移)
        2. 保存相对时间 (相对于全局 _time_origin)
        3. 这样可以正确匹配 camera_idx=0 -> lidar_idx=228
        """
        frames_dir = self._resolve_optional_path(dataset_root, self.config.lidar_frames_path)
        if frames_dir is None:
            raise FileNotFoundError("LiDAR frames directory is not configured or does not exist.")
        frame_paths = sorted(p for p in Path(frames_dir).glob("*.pcd") if p.is_file())
        if not frame_paths:
            raise FileNotFoundError(f"No .pcd files found under {frames_dir}")
        
        raw_times = self._load_lidar_timestamps(dataset_root)
        if not raw_times:
            raise FileNotFoundError("LiDAR timestamps are required when loading dynamic LiDAR scans.")
        if len(raw_times) != len(frame_paths):
            CONSOLE.log(
                f"[yellow]LiDAR timestamps ({len(raw_times)}) do not match frame count ({len(frame_paths)}); truncating."
            )
            count = min(len(raw_times), len(frame_paths))
            raw_times = raw_times[:count]
            frame_paths = frame_paths[:count]
        
        camera_abs_times = getattr(self, "_camera_absolute_times", None)
        camera_rel_times = getattr(self, "_camera_relative_times", None)
        camera_to_worlds = getattr(self, "_camera_to_world_tensor", None)

        # Fallback: 如果没有相机时间戳,用旧方法(插值)
        if not camera_abs_times or camera_rel_times is None or camera_to_worlds is None:
            CONSOLE.log("[yellow]Camera timing info unavailable, using interpolation-based alignment")
            aligned_times = self._align_lidar_times(raw_times)
            lidar_to_camera = getattr(self, "_lidar_to_camera_pose", None)
            if lidar_to_camera is None:
                lidar_to_camera = self._load_lidar_to_camera_transform(dataset_root)
                self._lidar_to_camera_pose = lidar_to_camera
            lidar_to_worlds: List[torch.Tensor] = []
            for time_value in aligned_times:
                pose = self._lookup_reference_pose(time_value)
                if pose is None:
                    if lidar_to_worlds:
                        pose = lidar_to_worlds[-1]
                    else:
                        pose = torch.eye(3, 4)
                l2w = torch.eye(4)
                l2w[:3, :4] = pose if pose.shape == (3, 4) else pose[:3, :4]
                l2w = torch.matmul(l2w, lidar_to_camera)
                lidar_to_worlds.append(l2w[:3, :4].clone())
            lidar_to_world_tensor = torch.stack([mat.float() for mat in lidar_to_worlds], dim=0)
            times_tensor = torch.tensor(aligned_times, dtype=torch.float64).unsqueeze(-1)
            zero_velocity = torch.zeros((len(aligned_times), 3), dtype=torch.float32)
            metadata = {
                "sensor_idxs": torch.zeros((len(aligned_times), 1), dtype=torch.int32),
                "linear_velocities_local": zero_velocity,
                "angular_velocities_local": zero_velocity,
                "timestamps": times_tensor,
            }
            lidars = Lidars(lidar_to_worlds=lidar_to_world_tensor, times=times_tensor, metadata=metadata)
            self._dynamic_lidar = True
            return lidars, frame_paths

        # 新方法: 时间戳最近邻匹配
        # 1. 填充LiDAR的绝对时间
        start_value = (
            raw_times[0]
            if raw_times and raw_times[0] is not None
            else float(camera_abs_times[0])
        )
        lidar_abs_times = self._fill_missing_lidar_times(raw_times, start_value)

        # 2. 为每个相机找最近的LiDAR帧 (用绝对时间匹配)
        cam_times_tensor = torch.tensor(camera_abs_times, dtype=torch.float64)
        lidar_times_tensor = torch.tensor(lidar_abs_times, dtype=torch.float64)
        time_diffs = torch.abs(cam_times_tensor[:, None] - lidar_times_tensor[None, :])
        nearest_lidar_idxs = torch.argmin(time_diffs, dim=1)

        # 3. 加载LiDAR外参
        lidar_to_camera = getattr(self, "_lidar_to_camera_pose", None)
        if lidar_to_camera is None:
            lidar_to_camera = self._load_lidar_to_camera_transform(dataset_root)
            self._lidar_to_camera_pose = lidar_to_camera
        lidar_to_camera = lidar_to_camera.float()

        # 4. 只保留被相机使用的LiDAR帧
        used_lidar_indices = sorted(
            set(idx for idx in nearest_lidar_idxs.tolist() if 0 <= idx < len(frame_paths))
        )
        matched_paths: List[Path] = []
        matched_lidar_to_world: List[torch.Tensor] = []
        matched_times: List[float] = []  # 相对时间!
        deltas: List[float] = []

        # 获取全局时间原点
        origin = getattr(self, "_time_origin", 0.0)

        for lidar_idx in used_lidar_indices:
            # 找最近的相机帧
            cam_idx = int(torch.argmin(time_diffs[:, lidar_idx]).item())
            cam_pose = camera_to_worlds[cam_idx]
            l2w = torch.eye(4, dtype=torch.float32)
            l2w[:3, :4] = cam_pose if cam_pose.shape == (3, 4) else cam_pose[:3, :4]
            l2w = torch.matmul(l2w, lidar_to_camera)
            matched_lidar_to_world.append(l2w[:3, :4].clone())
            matched_paths.append(frame_paths[lidar_idx])
            
            # 关键修复: 保存LiDAR的相对时间 (相对于全局原点)
            # 而不是相机的相对时间!
            matched_times.append(lidar_abs_times[lidar_idx] - origin)
            
            deltas.append(float(time_diffs[cam_idx, lidar_idx]))

        if not matched_paths:
            raise RuntimeError("Failed to match any LiDAR frames to camera timestamps.")

        if len(used_lidar_indices) < len(frame_paths):
            CONSOLE.log(
                f"[yellow]Using {len(matched_paths)} LiDAR frames matched to cameras; "
                f"dropped {len(frame_paths) - len(used_lidar_indices)} unmatched frames based on timestamps."
            )
        
        min_match, max_match = used_lidar_indices[0], used_lidar_indices[-1]
        CONSOLE.log(
            f"[green]Timestamp matching: first camera uses LiDAR frame {min_match}, last uses {max_match}; "
            f"mean Δt={np.mean(deltas):.4f}s, max Δt={np.max(deltas):.4f}s"
        )

        lidar_to_world_tensor = torch.stack([mat.float() for mat in matched_lidar_to_world], dim=0)
        times_tensor = torch.tensor(matched_times, dtype=torch.float64).unsqueeze(-1)
        zero_velocity = torch.zeros((len(matched_paths), 3), dtype=torch.float32)
        metadata = {
            "sensor_idxs": torch.zeros((len(matched_paths), 1), dtype=torch.int32),
            "linear_velocities_local": zero_velocity,
            "angular_velocities_local": zero_velocity,
            "timestamps": times_tensor,
        }
        lidars = Lidars(lidar_to_worlds=lidar_to_world_tensor, times=times_tensor, metadata=metadata)
        self._dynamic_lidar = True
        return lidars, matched_paths

    def _load_pcd_tensor(self, path: Path) -> torch.Tensor:
        xyz = self._parse_pcd_xyz(path).astype(np.float32)
        intensity = np.ones((xyz.shape[0], 1), dtype=np.float32)
        relative_times = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        data = np.concatenate([xyz, intensity, relative_times], axis=1)
        return torch.from_numpy(data)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        dataset_root = Path(self.config.data).expanduser()
        model_root = _resolve_dataset_path(dataset_root, self.config.colmap_model_path)
        self._dataset_root = dataset_root
        self._model_root = model_root
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

        image_timestamp_table = self._load_image_timestamp_table(dataset_root)
        frame_sensor_names: List[str] = []
        raw_timestamps: List[Optional[float]] = []
        mask_root = self._resolve_optional_path(dataset_root, self.config.masks_path) if self.config.masks_path else None
        mask_filenames: List[Optional[Path]] = []
        images_root_resolved = images_root.resolve()

        for idx, image in enumerate(cam_extrinsics):
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
            pose_tensor = torch.from_numpy(pose[:3, :4]).float()
            poses.append(pose_tensor)
            image_filenames.append(img_path)
            camera_ids.append(image.camera_id)
            sensor_idx = _sensor_idx(image.camera_id)
            sensor_idxs.append(sensor_idx)
            rel_name = Path(image.name)
            current_sensor_prefix = rel_name.parts[0] if len(rel_name.parts) > 1 else rel_name.stem.split("_")[0]
            frame_sensor_names.append(current_sensor_prefix)
            timestamp_key = rel_name.as_posix()
            raw_time = image_timestamp_table.get(timestamp_key)
            if raw_time is None:
                raw_time = image_timestamp_table.get(rel_name.name)
            raw_timestamps.append(raw_time)
            if mask_root is not None:
                try:
                    img_rel_to_images = img_path.resolve().relative_to(images_root_resolved)
                except Exception:
                    img_rel_to_images = rel_name
                mask_rel_name = img_rel_to_images
                if len(mask_rel_name.parts) == 1 and current_sensor_prefix:
                    mask_rel_name = Path(current_sensor_prefix) / mask_rel_name.name
                mask_path = self._resolve_mask_file(dataset_root, mask_root, mask_rel_name)
                if idx < 10 or mask_path is None:
                    rel_log = img_rel_to_images if isinstance(img_rel_to_images, Path) else Path(str(img_rel_to_images))
                    CONSOLE.log(
                        f"[cyan]Mask match img_idx={idx}: {rel_log} (sensor={current_sensor_prefix}) -> "
                        f"{mask_path.relative_to(dataset_root) if mask_path else 'NOT FOUND'}"
                    )
                mask_filenames.append(mask_path if mask_path is not None else None)

        if not poses:
            raise RuntimeError("No COLMAP frames were loaded. Check filters and paths.")

        self.config.cameras = tuple(sensor_names)

        camera_to_world_tensor = torch.stack(poses, dim=0)
        times = self._finalize_camera_times(raw_timestamps, frame_sensor_names, camera_to_world_tensor)
        self._camera_positions_world = camera_to_world_tensor[:, :3, 3].cpu().numpy()
        self._camera_to_world_tensor = camera_to_world_tensor  # 保存用于LiDAR对齐
        
        cameras = Cameras(
            camera_to_worlds=camera_to_world_tensor,
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
                "timestamps": times,
            },
        )
        
        if mask_root is not None and len(mask_filenames) == len(image_filenames):
            exists_count = len([m for m in mask_filenames if m is not None and m.is_file()])
            if exists_count == len(image_filenames):
                # All masks exist and are files; cast away Nones
                self._mask_filenames = [m for m in mask_filenames if m is not None]
                CONSOLE.log(f"[green]Loaded {len(self._mask_filenames)} masks from {mask_root}")
                # Log a few examples to verify camera-folder alignment (mirrors 3dgrut logic)
                sample_log = []
                for img_path, m_path in zip(image_filenames[:3], self._mask_filenames[:3]):
                    img_rel = img_path.relative_to(dataset_root) if img_path.is_absolute() else img_path
                    mask_rel = m_path.relative_to(dataset_root) if m_path.is_absolute() else m_path
                    sample_log.append(f"{img_rel} -> {mask_rel}")
                if sample_log:
                    CONSOLE.log("[green]Mask mapping samples: " + "; ".join(sample_log))
            else:
                self._mask_filenames = None
                CONSOLE.log(
                    f"[yellow]Masks requested at {mask_root} but not all were found "
                    f"(matched {exists_count}/{len(image_filenames)}); continuing without masks."
                )
        else:
            self._mask_filenames = None
        return cameras, image_filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        dataset_root = getattr(self, "_dataset_root", Path(self.config.data).expanduser())
        model_root = getattr(self, "_model_root", _resolve_dataset_path(dataset_root, self.config.colmap_model_path))
        if self.config.lidar_frames_path and self.config.lidar_timestamps_path:
            return self._load_dynamic_lidars(dataset_root)
        self._dynamic_lidar = False
        point_cloud = self._load_point_cloud(dataset_root, model_root)
        lidar_to_world = torch.eye(4, dtype=torch.float32)[:3].unsqueeze(0)
        lidars = Lidars(
            lidar_to_worlds=lidar_to_world,
            times=torch.zeros((1, 1), dtype=torch.float64),
            metadata={"sensor_idxs": torch.zeros((1, 1), dtype=torch.int32)},
        )
        source_path = getattr(self, "_point_cloud_source_path", model_root / "points3D.txt")
        return lidars, [source_path]

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[Tensor]:
        if getattr(self, "_dynamic_lidar", False):
            return [self._load_pcd_tensor(path) for path in filenames]
        dataset_root = getattr(self, "_dataset_root", Path(self.config.data).expanduser())
        model_root = getattr(self, "_model_root", _resolve_dataset_path(dataset_root, self.config.colmap_model_path))
        point_cloud = self._load_point_cloud(dataset_root, model_root)
        return [point_cloud]

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
        return images_root / rel_name

    def _resolve_mask_file(self, dataset_root: Path, masks_root: Path, rel_name: Path) -> Optional[Path]:
        """Strictly mirror image path under masks_root: same dirs, same stem, prefer .jpg.png."""
        if rel_name.is_absolute():
            rel_name = rel_name.relative_to(rel_name.anchor)
        stem_with_dir = rel_name.with_suffix("")  # drop current suffix for consistency
        candidates: List[Path] = [
            masks_root / stem_with_dir.with_suffix(".jpg.png"),  # strict: original.jpg -> original.jpg.png
            masks_root / stem_with_dir.with_suffix(".png"),
            masks_root / stem_with_dir.with_suffix(".jpg"),
        ]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            if resolved.exists() and resolved.is_file() and str(resolved).startswith(str(masks_root.resolve())):
                return resolved
        return None

    def _load_point_cloud(self, dataset_root: Path, model_root: Path) -> Tensor:
        if getattr(self, "_cached_point_cloud_tensor", None) is not None:
            return self._cached_point_cloud_tensor
        points_path = self._resolve_point_cloud_path(dataset_root, model_root)
        self._point_cloud_source_path = points_path
        positions, colors, errors = self._read_point_cloud_arrays(points_path)
        positions, colors, errors = self._sanitize_point_cloud_arrays(positions, colors, errors)
        positions, colors, errors = self._maybe_downsample_points(positions, colors, errors)
        used_fallback = False
        if positions.shape[0] < self.config.min_point_cloud_points:
            CONSOLE.log(
                f"[yellow]WARNING: COLMAP point cloud has only {positions.shape[0]} points "
                f"(< {self.config.min_point_cloud_points}), generating fallback"
            )
            positions, colors, errors = self._generate_fallback_point_cloud(points_path)
            used_fallback = True
        intensities = (colors.mean(axis=1, keepdims=True) / 255.0).astype(np.float32)
        times = np.zeros((positions.shape[0], 1), dtype=np.float32)
        point_cloud = np.concatenate([positions.astype(np.float32), intensities, times], axis=1)
        tensor = torch.from_numpy(point_cloud)
        finite_mask = torch.isfinite(tensor).all(dim=1)
        tensor = tensor[finite_mask]
        if tensor.numel() == 0:
            CONSOLE.log("[red]CRITICAL: Point cloud is empty after sanitization, using fallback")
            positions, colors, errors = self._generate_fallback_point_cloud(points_path)
            used_fallback = True
            intensities = (colors.mean(axis=1, keepdims=True) / 255.0).astype(np.float32)
            times = np.zeros((positions.shape[0], 1), dtype=np.float32)
            tensor = torch.from_numpy(np.concatenate([positions.astype(np.float32), intensities, times], axis=1))
            finite_mask = torch.isfinite(tensor).all(dim=1)
        CONSOLE.log(
            f"[green]Seed cloud ready: {tensor.shape[0]} points "
            f"({'fallback' if used_fallback else 'COLMAP'}) from '{points_path.name}'"
        )
        self._cached_point_cloud_tensor = tensor
        # Preserve sanitized COLMAP seeds for SFM-based initialization.
        self._points3d_positions = tensor[:, :3].clone()
        colors_tensor = torch.from_numpy(colors.astype(np.float32))
        if colors_tensor.shape[0] == finite_mask.shape[0]:
            colors_tensor = colors_tensor[finite_mask]
        elif colors_tensor.shape[0] != tensor.shape[0]:
            colors_tensor = colors_tensor[: tensor.shape[0]]
        self._points3d_colors = colors_tensor
        self._points3d_times = torch.zeros((self._points3d_positions.shape[0], 1), dtype=torch.float32)
        return tensor

    def _resolve_point_cloud_path(self, dataset_root: Path, model_root: Path) -> Path:
        if self.config.points3d_path:
            candidate = _resolve_dataset_path(dataset_root, self.config.points3d_path)
            if not candidate.exists():
                raise FileNotFoundError(f"Configured points3D path does not exist: {candidate}")
            return candidate
        prefer_binary = (
            self.config.use_binary_points
            if self.config.use_binary_points is not None
            else self.config.use_binary_model
        )
        candidates: List[Path] = []
        if prefer_binary:
            candidates.extend([model_root / "points3D.bin", model_root / "points3D.txt"])
        else:
            candidates.extend([model_root / "points3D.txt", model_root / "points3D.bin"])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing COLMAP points3D file under {model_root}")

    def _read_point_cloud_arrays(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".bin":
            positions, colors, errors = read_colmap_points3D_binary(str(path))
        elif suffix == ".txt":
            positions, colors, errors = read_colmap_points3D_text(str(path))
        else:
            raise ValueError(f"Unsupported COLMAP point cloud format: {path}")
        return positions, colors, errors

    def _maybe_downsample_points(self, positions: np.ndarray, colors: np.ndarray, errors: np.ndarray):
        max_points = self.config.max_point_cloud_points
        if max_points is None or max_points <= 0 or len(positions) <= max_points:
            return self._sanitize_point_cloud_arrays(positions, colors, errors)
        method = self.config.pointcloud_downsample_method.lower()
        if method == "farthest":
            positions, colors, errors = sample_points3d_data(positions, colors, errors, max_points)
            return self._sanitize_point_cloud_arrays(positions, colors, errors)
        if method == "random":
            indices = np.random.choice(len(positions), max_points, replace=False)
            return self._sanitize_point_cloud_arrays(positions[indices], colors[indices], errors[indices])
        raise ValueError(f"Unknown pointcloud_downsample_method: {self.config.pointcloud_downsample_method}")

    def _sanitize_point_cloud_arrays(
        self, positions: np.ndarray, colors: np.ndarray, errors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isfinite(positions).all(axis=1)
        mask &= np.isfinite(colors).all(axis=1)
        if errors is not None and errors.size:
            mask &= np.isfinite(errors).all(axis=1)
        positions = positions[mask]
        colors = colors[mask]
        errors = errors[mask]
        return positions, colors, errors

    def _generate_fallback_point_cloud(self, source_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        CONSOLE.print(
            f"[yellow]COLMAP point cloud '{source_path}' is empty or invalid. "
            f"Generating {self.config.fallback_pointcloud_size} synthetic seeds."
        )
        num_points = max(self.config.fallback_pointcloud_size, self.config.min_point_cloud_points)
        if hasattr(self, "_camera_positions_world") and len(self._camera_positions_world):
            mins = self._camera_positions_world.min(axis=0)
            maxs = self._camera_positions_world.max(axis=0)
        else:
            mins = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            maxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        padding = float(self.config.fallback_pointcloud_padding)
        mins = (mins - padding).astype(np.float32)
        maxs = (maxs + padding).astype(np.float32)
        positions = np.random.uniform(mins, maxs, size=(num_points, 3)).astype(np.float32)
        colors = (np.random.rand(num_points, 3) * 255.0).astype(np.float32)
        errors = np.zeros((num_points, 1), dtype=np.float32)
        return positions, colors, errors

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

    def _generate_dataparser_outputs(self, split="train"):
        """Augment base outputs with COLMAP SFM seeds while still loading LiDAR."""
        outputs = super()._generate_dataparser_outputs(split=split)

        points = getattr(self, "_points3d_positions", None)
        colors = getattr(self, "_points3d_colors", None)
        if points is not None and colors is not None:
            points = points.clone()
            colors = colors.clone()
            if colors.shape[0] != points.shape[0]:
                colors = colors[: points.shape[0]]
            times = getattr(self, "_points3d_times", None)
            if times is None or times.shape[0] != points.shape[0]:
                times = torch.zeros((points.shape[0], 1), dtype=points.dtype)

            transform = outputs.dataparser_transform
            if transform is not None and transform.numel():
                transform = transform.to(points.device)
                if transform.shape == (3, 4):
                    transform = torch.cat(
                        [
                            transform,
                            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=transform.device, dtype=transform.dtype),
                        ],
                        dim=0,
                    )
                if transform.shape == (4, 4):
                    points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
                    points = (transform @ points_h.T).T[:, :3]

            outputs.metadata["points3D_xyz"] = points
            outputs.metadata["points3D_rgb"] = colors
            outputs.metadata["points3D_times"] = times.to(points.device)
        return outputs
