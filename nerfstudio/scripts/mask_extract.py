#!/usr/bin/env python3
"""
Extract masks by rendering only the gaussians selected by input masks.
Multi-frame voting + filtering by MIN_FRAME_RATIO.
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Set

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nerfstudio.utils.eval_utils import eval_setup

_CAPTURED_INFO = {}


def setup_rasterization_capture() -> bool:
    """Monkey patch rasterization to capture info."""
    global _CAPTURED_INFO
    try:
        import gsplat_original.rendering as gsplat_rendering

        original_rasterization = gsplat_rendering.rasterization

        def patched_rasterization(*args, **kwargs):
            render, alpha, info = original_rasterization(*args, **kwargs)
            _CAPTURED_INFO["last_info"] = info
            _CAPTURED_INFO["last_render"] = render
            _CAPTURED_INFO["last_alpha"] = alpha
            return render, alpha, info

        gsplat_rendering.rasterization = patched_rasterization
        print("[setup] Rasterization capture enabled (gsplat_original)")
        return True
    except ImportError:
        pass

    try:
        import gsplat

        original_rasterization = gsplat.rasterization

        def patched_rasterization(*args, **kwargs):
            render, alpha, info = original_rasterization(*args, **kwargs)
            _CAPTURED_INFO["last_info"] = info
            _CAPTURED_INFO["last_render"] = render
            _CAPTURED_INFO["last_alpha"] = alpha
            return render, alpha, info

        gsplat.rasterization = patched_rasterization
        print("[setup] Rasterization capture enabled (gsplat)")
        return True
    except ImportError:
        pass

    print("[setup] WARNING: Could not patch rasterization - info capture disabled")
    return False


def get_captured_info():
    return _CAPTURED_INFO.get("last_info"), _CAPTURED_INFO.get("last_render"), _CAPTURED_INFO.get("last_alpha")


def clear_captured_info():
    _CAPTURED_INFO.clear()


def _override_config(cfg, ckpt_path: Path):
    cfg.load_dir = ckpt_path.parent
    name = ckpt_path.name
    try:
        step_str = name.split("-")[1].split(".")[0]
        cfg.load_step = int(step_str)
    except Exception:
        cfg.load_step = None
    if hasattr(cfg.pipeline, "datamanager"):
        cfg.pipeline.datamanager.max_thread_workers = 0
        if hasattr(cfg.pipeline.datamanager, "num_processes"):
            cfg.pipeline.datamanager.num_processes = 0
    if hasattr(cfg.pipeline, "model") and hasattr(cfg.pipeline.model, "use_ray_tracing"):
        cfg.pipeline.model.use_ray_tracing = False
    return cfg


def extract_real_frame_idx(
    meta: dict, fallback: int, image_filename: str = None, image_name: str = None
) -> int:
    """Extract real frame index from metadata or filenames."""
    for k in ["frame_num", "frame_idx", "frame_id", "image_idx", "timestamp_idx"]:
        if k in meta:
            v = meta[k]
            if isinstance(v, torch.Tensor):
                return int(v.item())
            return int(v)

    if "filepath" in meta:
        filepath = meta["filepath"]
        if isinstance(filepath, (list, tuple)):
            filepath = filepath[0]
        match = re.search(r"[/\\]?(\d+)\.png", str(filepath))
        if match:
            return int(match.group(1))

    if "image_name" in meta:
        name = meta["image_name"]
        match = re.search(r"(\d+)", str(name))
        if match:
            return int(match.group(1))

    if image_filename:
        stem = Path(str(image_filename)).stem
        match = re.search(r"(\d+)", stem)
        if match:
            return int(match.group(1))

    if image_name:
        stem = Path(str(image_name)).name
        match = re.search(r"(\d+)", stem)
        if match:
            return int(match.group(1))

    return fallback


def _load_mask(mask_path: Path, size: tuple) -> torch.Tensor:
    """Load and resize mask to target size."""
    img = Image.open(mask_path).convert("L").resize(size, resample=Image.NEAREST)
    arr = np.array(img)
    return torch.from_numpy(arr >= 1)


def _tiles_from_mask(mask: torch.Tensor, tile_size: int, tile_width: int) -> torch.Tensor:
    """Convert mask to a flat list of tile indices."""
    H, W = mask.shape
    tile_h = (H + tile_size - 1) // tile_size
    tile_w = tile_width
    pad_h = max(0, tile_h * tile_size - H)
    pad_w = max(0, tile_w * tile_size - W)

    m = mask
    if pad_h > 0 or pad_w > 0:
        m = F.pad(m, (0, pad_w, 0, pad_h))
    m = m.view(tile_h, tile_size, tile_w, tile_size)
    tile_hit = m.any(dim=3).any(dim=1)
    tiles = tile_hit.flatten().nonzero(as_tuple=False).squeeze(1)
    return tiles


def _vote_gaussians_from_mask(
    mask: torch.Tensor,
    info: dict,
    num_gauss_total: int,
    verbose: bool = False,
) -> Set[int]:
    """Vote gaussians intersecting mask tiles."""
    if info is None:
        raise ValueError("info is None - cannot perform voting!")

    isect_offsets = info["isect_offsets"].reshape(-1)
    flatten_ids = info["flatten_ids"]
    tile_size = int(info["tile_size"].item() if isinstance(info["tile_size"], torch.Tensor) else info["tile_size"])
    tile_width = int(info["tile_width"].item() if isinstance(info["tile_width"], torch.Tensor) else info["tile_width"])

    if verbose:
        print(
            "    [vote_debug] "
            f"flatten_ids shape={flatten_ids.shape}, num_gauss_total={num_gauss_total}, "
            f"tile_size={tile_size}, tile_width={tile_width}"
        )

    tiles = _tiles_from_mask(mask, tile_size, tile_width)
    if tiles.numel() == 0:
        if verbose:
            print("    [vote_debug] No tiles hit by mask")
        return set()

    visible_ids = None
    if "gaussian_ids" in info and info["gaussian_ids"] is not None:
        gids = info["gaussian_ids"]
        if isinstance(gids, torch.Tensor) and gids.ndim == 2:
            gids = gids[0]
        visible_ids = gids if isinstance(gids, torch.Tensor) else torch.as_tensor(gids, device=flatten_ids.device)
    elif "tiles_per_gauss" in info and info["tiles_per_gauss"] is not None:
        tpg = info["tiles_per_gauss"]
        if isinstance(tpg, torch.Tensor):
            tpg = tpg.reshape(-1, tpg.shape[-1])
            vis_mask = tpg.any(dim=0)
            visible_ids = vis_mask.nonzero(as_tuple=False).squeeze(1)

    if visible_ids is None:
        print("    [vote_warning] No gaussian_ids in info, using all gaussians")
        visible_ids = torch.arange(num_gauss_total, device=flatten_ids.device)

    device = visible_ids.device
    vis_flag = torch.zeros((num_gauss_total,), dtype=torch.bool, device=device)
    vis_flag[visible_ids] = True

    frame_gaussians: Set[int] = set()
    for tile_idx in tiles.tolist():
        start = int(isect_offsets[tile_idx].item())
        end = int(isect_offsets[tile_idx + 1].item()) if tile_idx + 1 < isect_offsets.numel() else int(flatten_ids.numel())
        if end <= start:
            continue

        candidates = (flatten_ids[start:end] % num_gauss_total).to(torch.long)
        visible = candidates[vis_flag[candidates]]
        if visible.numel() > 0:
            frame_gaussians.update(map(int, visible.tolist()))

    if verbose:
        print(f"    [vote_debug] Unique gaussians voted: {len(frame_gaussians)}")

    return frame_gaussians


def _save_mask_image(mask_tensor: torch.Tensor, output_path: Path) -> None:
    """Save RGB render as grayscale mask."""
    mask_tensor = mask_tensor.detach()
    
    if mask_tensor.dim() == 4:
        img_tensor = mask_tensor[0]
    elif mask_tensor.dim() == 3:
        if mask_tensor.shape[0] == 3:
            img_tensor = mask_tensor.permute(1, 2, 0)
        else:
            img_tensor = mask_tensor
    else:
        raise ValueError(f"Unexpected shape: {mask_tensor.shape}")
    
    gray = img_tensor.max(dim=-1)[0]
    img_np = (gray.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img_np).save(output_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", choices=["eval", "train"], default="train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--target-cam-idx", type=int, default=0)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--output-mask-root", type=Path, required=True)
    ap.add_argument("--min-frame-ratio", type=float, default=1)

    args = ap.parse_args()

    device = torch.device(args.device)
    args.output_mask_root.mkdir(parents=True, exist_ok=True)

    capture_enabled = setup_rasterization_capture()
    if not capture_enabled:
        print("[error] Cannot enable masking without rasterization capture!")
        sys.exit(1)

    print("[setup] Loading pipeline...")
    config, pipeline, _, _ = eval_setup(
        args.config,
        test_mode="inference",
        update_config_callback=lambda cfg: _override_config(cfg, args.checkpoint),
    )
    pipeline.to(device)
    model = pipeline.model
    model.eval()

    if args.split == "eval":
        dataset = pipeline.datamanager.eval_dataset
        dataparser_outputs = getattr(pipeline.datamanager, "eval_dataparser_outputs", None)
    else:
        dataset = pipeline.datamanager.train_dataset
        dataparser_outputs = getattr(pipeline.datamanager, "train_dataparser_outputs", None)
    if dataparser_outputs is None:
        dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
    cameras_all = dataset.cameras.to(device)
    images_root = None
    if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
        images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))

    if not hasattr(model, "features_dc"):
        raise AttributeError("Model has no features_dc field")
    if not hasattr(model, "features_rest"):
        raise AttributeError("Model has no features_rest field")

    num_gauss = model.means.shape[0]
    original_dc = model.features_dc.data.clone()
    original_rest = model.features_rest.data.clone()

    # ========== 阶段1: 投票统计 ==========
    print(f"\n{'='*60}")
    print("[Phase 1] Voting across frames...")
    gaussian_frame_count = torch.zeros(num_gauss, dtype=torch.int32, device=device)
    
    voted_frames = []
    missing_mask_count = 0
    
    for idx in range(len(cameras_all)):
        cam = cameras_all[idx : idx + 1]
        meta = cam.metadata or {}

        cam_idx = None
        for k in ["sensor_idxs", "sensor_idx", "cam_idx", "camera_idx", "camera_id"]:
            if k in meta:
                v = meta[k]
                cam_idx = int(v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                break
        if cam_idx is None or cam_idx != args.target_cam_idx:
            continue

        if args.max_frames is not None and len(voted_frames) >= args.max_frames:
            break

        image_filename = None
        image_name = None
        if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
            if idx < len(dataparser_outputs.image_filenames):
                image_filename = dataparser_outputs.image_filenames[idx]
                if images_root is not None:
                    try:
                        image_name = str(Path(image_filename).with_suffix("").relative_to(images_root))
                    except ValueError:
                        image_name = str(Path(image_filename).with_suffix("").name)
                else:
                    image_name = str(Path(image_filename).with_suffix("").name)

        real_frame_idx = extract_real_frame_idx(meta, len(voted_frames), image_filename=image_filename, image_name=image_name)

        clear_captured_info()
        with torch.no_grad():
            _ = model.get_outputs_for_camera(cam)
        info, _, _ = get_captured_info()
        if info is None:
            continue

        mask_name = f"car_mask_001/id1_{real_frame_idx:02d}_mask.png"
        mask_path = args.mask_root / mask_name
        if not mask_path.exists():
            if missing_mask_count < 5:
                print(f"  [skip] Missing mask: {mask_path}")
            missing_mask_count += 1
            continue

        W, H = int(cam.width.item()), int(cam.height.item())
        mask = _load_mask(mask_path, size=(W, H)).to(device)
        if mask.sum() == 0:
            continue

        frame_gauss_set = _vote_gaussians_from_mask(mask, info, num_gauss_total=num_gauss, verbose=(len(voted_frames) < 3))
        if len(frame_gauss_set) > 0:
            gauss_tensor = torch.tensor(list(frame_gauss_set), dtype=torch.long, device=device)
            gaussian_frame_count[gauss_tensor] += 1
            voted_frames.append((idx, real_frame_idx, cam))
            print(f"  [vote {len(voted_frames)}] frame={real_frame_idx}, voted={len(frame_gauss_set)} gaussians")

    print(f"\n{'='*60}")
    print(f"[Phase 1 Results]")
    print(f"  Total frames voted: {len(voted_frames)}")
    print(f"  Missing masks: {missing_mask_count}")
    
    # ========== 阶段2: 筛选高斯 ==========
    min_frames = int(len(voted_frames) * args.min_frame_ratio)
    selected_mask = gaussian_frame_count >= min_frames
    num_selected = int(selected_mask.sum().item())
    
    print(f"\n[Phase 2] Filtering gaussians...")
    print(f"  Min frame threshold: {min_frames} (ratio={args.min_frame_ratio})")
    print(f"  Selected gaussians: {num_selected} / {num_gauss} ({100*num_selected/num_gauss:.3f}%)")
    
    if num_selected == 0:
        print("\n⚠️  No gaussians selected! Try lower --min-frame-ratio")
        return
    
    nonzero_mask = gaussian_frame_count > 0
    num_nonzero = int(nonzero_mask.sum().item())
    if num_nonzero > 0:
        nonzero_counts = gaussian_frame_count[nonzero_mask]
        print(f"  Gaussians with ANY appearance: {num_nonzero}")
        print(f"    Range: {nonzero_counts.min().item()}-{nonzero_counts.max().item()} frames")
        print(f"    Mean: {nonzero_counts.float().mean().item():.2f} frames")
    
    selected_ids = selected_mask.nonzero(as_tuple=False).squeeze(1)
    
    # ========== 阶段3: 渲染 masked 图像 ==========
    print(f"\n{'='*60}")
    print(f"[Phase 3] Rendering {len(voted_frames)} masked images...")
    
    rendered = 0
    for idx, real_frame_idx, cam in voted_frames:
        with torch.no_grad():
            # 全黑
            model.features_dc.data[:] = -255.0
            model.features_rest.data[:] = 0.0
            
            # 选中的变白
            model.features_dc.data[selected_ids] = 255.0
            model.features_rest.data[selected_ids] = 0.0
            
            # 渲染
            outputs = model.get_outputs_for_camera(cam)
            
            # 恢复
            model.features_dc.data.copy_(original_dc)
            model.features_rest.data.copy_(original_rest)

        if "rgb" not in outputs:
            continue

        out_name = f"00{real_frame_idx:02d}_mask.png"
        out_path = args.output_mask_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        _save_mask_image(outputs["rgb"], out_path)
        if rendered < 5:
            print(f"  [save] {out_path}")
        
        rendered += 1

    print(f"\n{'='*60}")
    print(f"✅ Done! Rendered {rendered} masks to {args.output_mask_root}")


if __name__ == "__main__":
    main()