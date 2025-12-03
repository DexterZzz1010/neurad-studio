#!/usr/bin/env python3
"""Project a LiDAR frame onto a COLMAP image using the exact colmap-splatgut dataparser setup."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import FullImageLidarDatamanagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig
from nerfstudio.utils.poses import inverse


def _build_dataparser(config_name: str, data_path: Path) -> ColmapDataParser:
    if config_name not in method_configs:
        raise KeyError(f"Unknown config '{config_name}'. Available: {', '.join(sorted(method_configs.keys()))}")
    trainer_cfg = deepcopy(method_configs[config_name])
    datamanager_cfg = trainer_cfg.pipeline.datamanager
    if not isinstance(datamanager_cfg, FullImageLidarDatamanagerConfig):
        raise TypeError(f"Config '{config_name}' does not use FullImageLidarDatamanagerConfig.")
    dataparser_cfg = deepcopy(datamanager_cfg.dataparser)
    if not isinstance(dataparser_cfg, ColmapDataParserConfig):
        raise TypeError(f"Config '{config_name}' is not using the COLMAP dataparser.")
    dataparser_cfg.data = data_path
    return dataparser_cfg.setup()


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Map depth to a simple blue->red gradient without external colormaps."""
    depth_min, depth_max = float(depth.min()), float(depth.max())
    if depth_max - depth_min < 1e-6:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    r = np.clip(depth_norm * 1.3, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(depth_norm - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip(1.2 - depth_norm * 1.2, 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _draw_points(image: np.ndarray, u: np.ndarray, v: np.ndarray, colors: np.ndarray, size: int) -> np.ndarray:
    """Draw small squares centered at (u, v)."""
    h, w, _ = image.shape
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            uu = u + dx
            vv = v + dy
            mask = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            image[vv[mask], uu[mask]] = colors[mask]
    return image


def project_lidar_to_image(
    data_path: Path,
    output_path: Path,
    config_name: str,
    split: str,
    camera_idx: int,
    max_points: int,
    point_size: int,
) -> None:
    torch.set_grad_enabled(False)
    dataparser = _build_dataparser(config_name, data_path)
    outputs = dataparser.get_dataparser_outputs(split=split)

    cameras = outputs.cameras
    metadata = outputs.metadata
    lidars = metadata["lidars"]
    point_clouds = metadata["point_clouds"]

    # Validate camera_idx
    if camera_idx < 0 or camera_idx >= len(outputs.image_filenames):
        raise IndexError(f"camera_idx {camera_idx} out of range (0, {len(outputs.image_filenames) - 1})")

    # === Automatic timestamp matching ===
    camera_time = float(cameras.times[camera_idx])
    
    # Calculate time differences
    time_diffs = torch.abs(lidars.times.squeeze(-1) - camera_time)
    lidar_idx = int(torch.argmin(time_diffs).item())
    delta_t = float(time_diffs[lidar_idx].item())
    
    # Print matching results
    print("\n" + "="*60)
    print("TIMESTAMP MATCHING:")
    print(f"  Camera index: {camera_idx}")
    print(f"  Camera time: {camera_time:.6f}s")
    print(f"  → Matched LiDAR index: {lidar_idx}")
    print(f"  → LiDAR time: {float(lidars.times[lidar_idx]):.6f}s")
    print(f"  → Time difference (Δt): {delta_t:.6f}s")
    if delta_t > 0.1:
        print(f"  ⚠️  Warning: Time diff > 100ms, alignment may be poor!")
    print("="*60 + "\n")
    
    # Validate lidar_idx
    if lidar_idx < 0 or lidar_idx >= len(point_clouds):
        raise IndexError(f"Matched lidar_idx {lidar_idx} out of range (0, {len(point_clouds) - 1})")

    # Trajectory statistics
    if cameras.times is not None:
        cam_times = cameras.times.squeeze(-1).cpu().numpy()
        if len(cam_times) > 1:
            cam_dt = np.diff(np.sort(cam_times))
            print(
                f"Camera trajectory: {len(cam_times)} frames, dt_mean={cam_dt.mean():.6f}s, "
                f"dt_min={cam_dt.min():.6f}s, dt_max={cam_dt.max():.6f}s"
            )
    
    if lidars.times is not None:
        lidar_times = lidars.times.squeeze(-1).cpu().numpy()
        if len(lidar_times) > 1:
            lidar_dt = np.diff(np.sort(lidar_times))
            print(
                f"LiDAR trajectory: {len(lidar_times)} frames, dt_mean={lidar_dt.mean():.6f}s, "
                f"dt_min={lidar_dt.min():.6f}s, dt_max={lidar_dt.max():.6f}s"
            )

    # Load image
    image_path = outputs.image_filenames[camera_idx]
    image = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = image.shape[:2]

    # Load point cloud
    point_cloud = point_clouds[lidar_idx]
    if max_points > 0 and point_cloud.shape[0] > max_points:
        idxs = torch.randperm(point_cloud.shape[0])[:max_points]
        point_cloud = point_cloud[idxs]

    # Apply velocity compensation if available
    velocities = lidars.metadata.get("linear_velocities_local")
    if velocities is not None and len(velocities) > lidar_idx:
        point_cloud = point_cloud.clone()
        point_cloud[:, :3] -= velocities[lidar_idx] * point_cloud[:, 4:5]

    # Get poses
    c2w = cameras.camera_to_worlds[camera_idx]
    l2w = lidars.lidar_to_worlds[lidar_idx]
    print("c2w_used:\n", c2w.cpu().numpy())
    print("l2w_used:\n", l2w.cpu().numpy())
    
    w2c = inverse(c2w)
    
    # Build 4x4 matrices
    pts = point_cloud[:, :3]
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N,4)
    
    l2w_4 = torch.eye(4, device=pts.device, dtype=pts.dtype)
    l2w_4[:3, :4] = l2w
    
    w2c_4 = torch.eye(4, device=pts.device, dtype=pts.dtype)
    w2c_4[:3, :4] = w2c
    
    # Coordinate system sanity check
    l2c_4 = w2c_4 @ l2w_4
    test_points = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # forward
            [0.0, 10.0, 0.0],  # left
            [0.0, 0.0, 10.0],  # up
        ],
        device=pts.device,
        dtype=pts.dtype,
    )
    test_h = torch.cat([test_points, torch.ones((3, 1), device=pts.device, dtype=pts.dtype)], dim=1)
    test_cam = (test_h @ l2c_4.T)[:, :3].cpu().numpy()
    
    print("\n" + "=" * 60)
    print("Coordinate system check (LiDAR frame -> Camera frame using current l2c):")
    print(f"  Forward (10,0,0) -> ({test_cam[0,0]:7.3f}, {test_cam[0,1]:7.3f}, {test_cam[0,2]:7.3f})")
    print(f"  Left    (0,10,0) -> ({test_cam[1,0]:7.3f}, {test_cam[1,1]:7.3f}, {test_cam[1,2]:7.3f})")
    print(f"  Up      (0,0,10) -> ({test_cam[2,0]:7.3f}, {test_cam[2,1]:7.3f}, {test_cam[2,2]:7.3f})")
    print("Expected for OpenCV RDF camera: +Z forward, +X right, +Y down")
    print("=" * 60 + "\n")

    # Transform points to camera frame
    pts_world = (pts_h @ l2w_4.T)[:, :3]
    points_cam = (torch.cat([pts_world, ones], dim=1) @ w2c_4.T)[:, :3]

    # Get camera intrinsics
    fx = float(cameras.fx[camera_idx])
    fy = float(cameras.fy[camera_idx])
    cx = float(cameras.cx[camera_idx])
    cy = float(cameras.cy[camera_idx])
    width = int(cameras.width[camera_idx])
    height = int(cameras.height[camera_idx])
    
    # Check image size consistency
    if (width != img_w) or (height != img_h):
        sx = img_w / width if width else float("nan")
        sy = img_h / height if height else float("nan")
        print(
            f"[WARN] Image size {img_w}x{img_h} differs from intrinsics {width}x{height}; "
            f"scale factors sx={sx:.4f}, sy={sy:.4f}"
        )

    # Fisheye projection (Kannala-Brandt model)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    # Filter points behind camera
    valid = z > 1e-6
    x, y, z = x[valid], y[valid], z[valid]

    # Kannala-Brandt projection
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(r, z)

    # Distortion coefficients for FLCW camera
    k1 = torch.tensor(-0.027404852, device=points_cam.device, dtype=points_cam.dtype)
    k2 = torch.tensor(0.028455389, device=points_cam.device, dtype=points_cam.dtype)
    k3 = torch.tensor(-0.025744684, device=points_cam.device, dtype=points_cam.dtype)
    k4 = torch.tensor(0.009067111, device=points_cam.device, dtype=points_cam.dtype)

    theta2 = theta * theta
    theta_d = theta * (1.0 + k1*theta2 + k2*theta2*theta2 + k3*theta2*theta2*theta2 + k4*theta2*theta2*theta2*theta2)

    # Project to pixel coordinates
    inv_r = torch.where(r > 1e-8, 1.0 / r, torch.zeros_like(r))
    u = fx * theta_d * x * inv_r + cx
    v = fy * theta_d * y * inv_r + cy

    # Convert to numpy
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    z_np = z.cpu().numpy()

    # Filter points outside image bounds
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_frame].astype(np.int64)
    v = v[in_frame].astype(np.int64)
    z_np = z_np[in_frame]
    
    # Generate colors and draw
    colors = _depth_to_rgb(z_np)
    # image = np.ones_like(image) * 255  # Uncomment to use white background
    overlay = _draw_points(image.copy(), u, v, colors, point_size)
    Image.fromarray(overlay).save(output_path)

    print(f"\nSaved overlay to {output_path}")
    print(
        f"camera_idx={camera_idx}, lidar_idx={lidar_idx}, Δt={delta_t:.6f}s, "
        f"projected_points={len(u)}/{len(valid)}, image={image_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project LiDAR points into a COLMAP image using the colmap-splatgut dataparser logic."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset root (same path used for training).")
    parser.add_argument(
        "--config-name",
        default="colmap-splatgut",
        help="Method config to pull dataparser defaults from (default: colmap-splatgut).",
    )
    parser.add_argument("--split", default="train", choices=["train", "eval", "test"], help="Split to load.")
    parser.add_argument(
        "--camera-idx",
        type=int,
        default=0,
        help="Camera index within the split; will automatically find closest LiDAR frame by timestamp.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Randomly subsample this many points before projection (<=0 disables subsampling).",
    )
    parser.add_argument("--point-size", type=int, default=1, help="Half-width in pixels for each drawn point.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_projection.png"),
        help="Where to write the projected overlay image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_lidar_to_image(
        data_path=args.data,
        output_path=args.output,
        config_name=args.config_name,
        split=args.split,
        camera_idx=args.camera_idx,
        max_points=args.max_points,
        point_size=args.point_size,
    )