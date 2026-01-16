#!/usr/bin/env python3
"""
修复版本：确保投票和渲染使用相同的坐标系
通过monkey patch获取get_outputs_for_camera内部的rasterization info
"""
import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import CameraType

# 全局变量用于捕获info
_CAPTURED_INFO = {}


# ============================================================================
# Monkey patch: 捕获rasterization info
# ============================================================================

def setup_rasterization_capture():
    """
    设置monkey patch来捕获rasterization的info
    """
    global _CAPTURED_INFO
    
    try:
        # 尝试导入gsplat_original
        import gsplat_original.rendering as gsplat_rendering
        original_rasterization = gsplat_rendering.rasterization
        
        def patched_rasterization(*args, **kwargs):
            render, alpha, info = original_rasterization(*args, **kwargs)
            # 捕获info到全局变量
            _CAPTURED_INFO['last_info'] = info
            _CAPTURED_INFO['last_render'] = render
            _CAPTURED_INFO['last_alpha'] = alpha
            return render, alpha, info
        
        # 替换函数
        gsplat_rendering.rasterization = patched_rasterization
        print("[setup] ✅ Rasterization capture enabled (gsplat_original)")
        return True
    
    except ImportError:
        pass
    
    try:
        # 尝试导入gsplat
        import gsplat
        original_rasterization = gsplat.rasterization
        
        def patched_rasterization(*args, **kwargs):
            render, alpha, info = original_rasterization(*args, **kwargs)
            _CAPTURED_INFO['last_info'] = info
            _CAPTURED_INFO['last_render'] = render
            _CAPTURED_INFO['last_alpha'] = alpha
            return render, alpha, info
        
        gsplat.rasterization = patched_rasterization
        print("[setup] ✅ Rasterization capture enabled (gsplat)")
        return True
    
    except ImportError:
        pass
    
    print("[setup] ⚠️  WARNING: Could not patch rasterization - info capture disabled")
    return False


def get_captured_info():
    """获取最近一次rasterization的info"""
    global _CAPTURED_INFO
    return _CAPTURED_INFO.get('last_info'), _CAPTURED_INFO.get('last_render'), _CAPTURED_INFO.get('last_alpha')


def clear_captured_info():
    """清空捕获的info"""
    global _CAPTURED_INFO
    _CAPTURED_INFO.clear()


# ============================================================================
# 原有的辅助函数
# ============================================================================

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


def _debug_cuda(tag: str, sync: bool):
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"[cuda] {tag}: used={used/1e9:.2f}GB free={free/1e9:.2f}GB total={total/1e9:.2f}GB")


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
        match = re.search(r'[/\\]?(\d+)\.png', str(filepath))
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


# ============================================================================
# Masking相关函数
# ============================================================================

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



# ============================================================================
# 在 Masking相关函数 部分添加
# ============================================================================

def _visualize_voting_result(
    frame_gauss_set: Set[int],
    info: dict,
    mask: torch.Tensor,
    outputs: dict,
    save_path: Path,
    frame_idx: int,
    device: torch.device,
):
    """
    可视化投票结果：绘制选中高斯的2D投影和mask边界
    """
    try:
        import cv2  # type: ignore
        _has_cv2 = True
    except Exception:
        cv2 = None
        _has_cv2 = False
    
    if len(frame_gauss_set) == 0:
        print(f"  [viz] Skipping visualization - no gaussians voted")
        return
    
    # 获取渲染图
    if "rgb" in outputs:
        render_img = outputs["rgb"]
    elif "image" in outputs:
        render_img = outputs["image"]
    elif "comp_rgb" in outputs:
        render_img = outputs["comp_rgb"]
    else:
        print(f"  [viz] No RGB output available")
        return

    # Detach to avoid autograd tracking when converting to numpy
    render_img = render_img.detach()
    
    # 转numpy [H, W, 3]
    if render_img.dim() == 4:  # [1, H, W, 3]
        render_np = render_img[0].cpu().numpy()
    elif render_img.dim() == 3:
        if render_img.shape[0] == 3:  # [3, H, W]
            render_np = render_img.permute(1, 2, 0).cpu().numpy()
        else:  # [H, W, 3]
            render_np = render_img.cpu().numpy()
    else:
        print(f"  [viz] Unexpected render shape: {render_img.shape}")
        return
    
    render_np = (render_np * 255).clip(0, 255).astype(np.uint8)
    H, W = render_np.shape[:2]
    
    # 创建可视化画布
    vis = np.array(render_np, dtype=np.uint8, copy=True, order="C")
    if vis.ndim == 2:
        vis = np.repeat(vis[..., None], 3, axis=2)
    elif vis.ndim == 3 and vis.shape[2] > 3:
        vis = vis[..., :3]
    
    # 1. 绘制投票的高斯（绿点）
    voted_ids = torch.tensor(list(frame_gauss_set), dtype=torch.long, device=device)
    
    if "means2d" not in info:
        print(f"  [viz] No means2d in info")
        return
    
    means2d = info["means2d"]
    means2d = means2d.detach()
    if means2d.ndim == 3:
        means2d = means2d[0]  # [N, 2]
    
    voted_means2d = means2d[voted_ids].cpu().numpy()
    
    # 随机采样（避免图像太密）
    max_points = 3000
    if len(voted_means2d) > max_points:
        sample_idx = np.random.choice(len(voted_means2d), max_points, replace=False)
        sampled_means2d = voted_means2d[sample_idx]
    else:
        sampled_means2d = voted_means2d
    
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    def _draw_with_pil() -> None:
        from PIL import ImageDraw, ImageFilter

        vis_img = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis_img)

        for pt in sampled_means2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        edges = mask_img.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(7))
        edge = np.asarray(edges) > 0

        vis_arr = np.asarray(vis_img).copy()
        vis_arr[edge] = (255, 0, 0)
        vis_img = Image.fromarray(vis_arr)

        draw = ImageDraw.Draw(vis_img)
        info_text = [
            f"Frame {frame_idx}",
            f"Voted: {len(frame_gauss_set)} gaussians",
            f"Shown: {len(sampled_means2d)} points",
            f"Mask: {mask.sum().item():.0f} pixels",
        ]
        y_offset = 10
        for line in info_text:
            draw.text((10, y_offset), line, fill=(255, 255, 0))
            y_offset += 18

        vis_img.save(save_path)

    if _has_cv2:
        try:
            for pt in sampled_means2d:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)  # 绿色

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 0, 0), 3)  # 红色粗线

            info_text = [
                f"Frame {frame_idx}",
                f"Voted: {len(frame_gauss_set)} gaussians",
                f"Shown: {len(sampled_means2d)} points",
                f"Mask: {mask.sum().item():.0f} pixels",
            ]

            y_offset = 30
            for line in info_text:
                cv2.putText(
                    vis,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 30

            legend_y = H - 80
            cv2.circle(vis, (20, legend_y), 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "Voted Gaussians",
                (35, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.line(vis, (10, legend_y + 30), (30, legend_y + 30), (255, 0, 0), 3)
            cv2.putText(
                vis,
                "Mask Boundary",
                (35, legend_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            Image.fromarray(vis).save(save_path)
        except Exception as e:
            print(f"  [viz] OpenCV draw failed ({type(e).__name__}: {e}); falling back to PIL")
            _draw_with_pil()
    else:
        _draw_with_pil()

    print(f"  [viz] 💾 Saved visualization: {save_path.name}")




def _vote_gaussians_from_mask(
    mask: torch.Tensor,
    info: dict,
    num_gauss_total: int,
    verbose: bool = False,
) -> Set[int]:
    """投票选出mask区域内的高斯"""
    if info is None:
        raise ValueError("info is None - cannot perform voting!")
    
    isect_offsets = info["isect_offsets"].reshape(-1)
    flatten_ids = info["flatten_ids"]
    tile_size = int(info["tile_size"].item() if isinstance(info["tile_size"], torch.Tensor) else info["tile_size"])
    tile_width = int(info["tile_width"].item() if isinstance(info["tile_width"], torch.Tensor) else info["tile_width"])

    if verbose:
        print(f"    [vote_debug] flatten_ids: shape={flatten_ids.shape}, range=[{flatten_ids.min().item()}, {flatten_ids.max().item()}]")
        print(f"    [vote_debug] num_gauss_total={num_gauss_total}")
        print(f"    [vote_debug] tile_size={tile_size}, tile_width={tile_width}")

    tiles = _tiles_from_mask(mask, tile_size, tile_width)
    if tiles.numel() == 0:
        if verbose:
            print("    [vote_debug] No tiles hit by mask")
        return set()

    # 获取visible IDs
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
        print("    [vote_warning] No gaussian_ids in info, using all gaussians (slow!)")
        visible_ids = torch.arange(num_gauss_total, device=flatten_ids.device)

    device = visible_ids.device
    vis_flag = torch.zeros((num_gauss_total,), dtype=torch.bool, device=device)
    vis_flag[visible_ids] = True

    frame_gaussians: Set[int] = set()
    total_candidates = 0
    total_visible_candidates = 0

    for tile_idx in tiles.tolist():
        start = int(isect_offsets[tile_idx].item())
        end = int(isect_offsets[tile_idx + 1].item()) if tile_idx + 1 < isect_offsets.numel() else int(flatten_ids.numel())
        if end <= start:
            continue

        candidates = (flatten_ids[start:end] % num_gauss_total).to(torch.long)
        total_candidates += len(candidates)

        visible = candidates[vis_flag[candidates]]
        total_visible_candidates += len(visible)

        if visible.numel() > 0:
            frame_gaussians.update(map(int, visible.tolist()))

    if verbose:
        print(f"    [vote_debug] Total candidates: {total_candidates}, visible: {total_visible_candidates}")
        print(f"    [vote_debug] Unique gaussians voted: {len(frame_gaussians)}")

    return frame_gaussians


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    
    ap.add_argument("--split", choices=["eval", "train"], default="train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--sync-cuda", action="store_true")
    
    # Masking参数
    ap.add_argument("--enable-masking", action="store_true")
    ap.add_argument("--mask-root", type=Path, default=None)
    ap.add_argument("--mask-template", default="car_mask_001/id1_{frame_idx:02d}_mask.png")
    ap.add_argument("--target-cam-idx", type=int, default=0)
    ap.add_argument("--min-frame-ratio", type=float, default=0.9)
    
    # 输出参数
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--render-output-root", type=Path, default=None)
    
    args = ap.parse_args()
    
    # 修复template
    if args.enable_masking:
        if '_mask.png' in args.mask_template and '02d_mask.png' in args.mask_template:
            print(f"[warning] Auto-fixing template format")
            args.mask_template = args.mask_template.replace(':02d_mask.png', ':02d}_mask.png')
    
    device = torch.device(args.device)

    # ========== 关键：设置monkey patch ==========
    capture_enabled = setup_rasterization_capture()
    if args.enable_masking and not capture_enabled:
        print("[error] Cannot enable masking without rasterization capture!")
        sys.exit(1)

    # ========== 加载模型 ==========
    print("[setup] Loading pipeline...")
    config, pipeline, _, step = eval_setup(
        args.config,
        test_mode="inference",
        update_config_callback=lambda cfg: _override_config(cfg, args.checkpoint),
    )
    pipeline.to(device)
    model = pipeline.model
    model.eval()

    print(f"✅ Loaded checkpoint from {args.checkpoint}")

    # ========== 准备数据 ==========
    if args.split == "eval":
        dataset = pipeline.datamanager.eval_dataset
        dataparser_outputs = getattr(pipeline.datamanager, "eval_dataparser_outputs", None)
    else:
        dataset = pipeline.datamanager.train_dataset
        dataparser_outputs = getattr(pipeline.datamanager, "train_dataparser_outputs", None)
    if dataparser_outputs is None:
        dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
    cameras_all = dataset.cameras.to(device)
    print(f"[info] Dataset: {len(cameras_all)} cameras")
    images_root = None
    if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
        images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))

    # 获取高斯总数
    if hasattr(model, "means"):
        num_gauss = model.means.shape[0]
    elif hasattr(model, "num_points"):
        num_gauss = model.num_points
    else:
        raise AttributeError("Cannot determine number of gaussians")
    
    print(f"[info] Total Gaussians: {num_gauss}")

    # ========== Masking初始化 ==========
    gaussian_frame_count = None
    missing_mask_count = 0
    cam_frame_counter = 0
    
    if args.enable_masking:
        if args.mask_root is None:
            raise ValueError("--enable-masking requires --mask-root")
        if not args.mask_root.exists():
            raise ValueError(f"Mask root not found: {args.mask_root}")
        
        gaussian_frame_count = torch.zeros(num_gauss, dtype=torch.int32, device=device)
        print(f"[masking] Enabled with mask_root={args.mask_root}")

    # ========== 循环：投票统计（使用monkey patched渲染）==========
    used = 0
    for idx in range(len(cameras_all)):
        cam = cameras_all[idx : idx + 1]
        meta = cam.metadata or {}

        # 提取cam_idx
        cam_idx = None
        for k in ["sensor_idxs", "sensor_idx", "cam_idx", "camera_idx", "camera_id"]:
            if k in meta:
                v = meta[k]
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    cam_idx = int(v.item())
                else:
                    cam_idx = int(v) if v is not None else None
                break

        if cam_idx is None or cam_idx != args.target_cam_idx:
            continue

        W, H = int(cam.width.item()), int(cam.height.item())
        print(f"\n[frame {used}] idx={idx} cam_idx={args.target_cam_idx} size={W}x{H}")

        # ========== 清空之前的捕获 ==========
        clear_captured_info()

        # ========== 渲染（会自动捕获info）==========
        _debug_cuda("before_render", args.sync_cuda)
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(cam)
        _debug_cuda("after_render", args.sync_cuda)

        # ========== 获取捕获的info ==========
        info, render, alpha = get_captured_info()
        
        if info is None:
            print(f"  [error] Failed to capture rasterization info!")
            if args.enable_masking:
                print(f"  [error] Cannot perform masking without info - skipping frame")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue
        else:
            if used < 3:
                print(f"  [debug] ✅ Captured info with keys: {list(info.keys())}")

        # ========== Masking投票 ==========
        if args.enable_masking:
            image_filename = None
            image_name = None
            if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
                if idx < len(dataparser_outputs.image_filenames):
                    image_filename = dataparser_outputs.image_filenames[idx]
                else:
                    print(
                        f"  [masking] image_filenames index out of range: idx={idx}, "
                        f"len={len(dataparser_outputs.image_filenames)}"
                    )
                if images_root is not None:
                    try:
                        image_name = str(Path(image_filename).with_suffix("").relative_to(images_root))
                    except ValueError:
                        image_name = str(Path(image_filename).with_suffix("").name)
                else:
                    image_name = str(Path(image_filename).with_suffix("").name)
            real_frame_idx = extract_real_frame_idx(
                meta, cam_frame_counter, image_filename=image_filename, image_name=image_name
            )
            if image_name:
                print(f"  [masking] image_name: {image_name}")
            else:
                print("  [masking] image_name: None (no dataparser image_filenames)")

            fmt_ctx = {}
            for k, v in meta.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    fmt_ctx[k] = v.item()
                elif not isinstance(v, torch.Tensor):
                    fmt_ctx[k] = v

            fmt_ctx["frame_idx"] = real_frame_idx
            fmt_ctx["cam_idx"] = cam_idx

            try:
                mask_name = args.mask_template.format(**fmt_ctx)
            except (KeyError, ValueError) as e:
                print(f"  [skip] Cannot format mask template: {e}")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

            mask_path = args.mask_root / mask_name

            if not mask_path.exists():
                if missing_mask_count < 5:
                    print(f"  [skip] Missing mask: {mask_path}")
                missing_mask_count += 1
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

            mask = _load_mask(mask_path, size=(W, H)).to(device)
            mask_pixel_count = int(mask.sum().item())
            
            if mask_pixel_count == 0:
                print(f"  [skip] Empty mask")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

            print(f"  [masking] ✅ Loaded mask: {mask_path.name}, pixels={mask_pixel_count}")

            # ✅ 投票
            frame_gauss_set = _vote_gaussians_from_mask(
                mask=mask,
                info=info,
                num_gauss_total=num_gauss,
                verbose=(used < 3),
            )

            if len(frame_gauss_set) > 0:
                gauss_tensor = torch.tensor(list(frame_gauss_set), dtype=torch.long, device=device)
                gaussian_frame_count[gauss_tensor] += 1
                print(f"  [masking] ✅ Voted {len(frame_gauss_set)} gaussians")
                
                # # ========== 可视化（前5帧）==========
                # if args.render_output_root is not None and used < 5:
                #     args.render_output_root.mkdir(parents=True, exist_ok=True)
                #     vis_path = args.render_output_root / f"debug_voted_frame{used:04d}.png"
                #     _visualize_voting_result(
                #         frame_gauss_set=frame_gauss_set,
                #         info=info,
                #         mask=mask,
                #         outputs=outputs,
                #         save_path=vis_path,
                #         frame_idx=used,
                #         device=device,
                #     )
            else:
                print(f"  [masking] ⚠️  No gaussians voted")
            cam_frame_counter += 1

        used += 1
        if used >= args.max_frames:
            break

    print(f"\n[done] Processed {used} frames")

    # ========== 循环结束：统计+mask+渲染 ==========
    if args.enable_masking and gaussian_frame_count is not None:
        min_frames = int(used * args.min_frame_ratio)
        selected = gaussian_frame_count >= min_frames
        num_selected = int(selected.sum().item())

        print(f"\n{'='*60}")
        print(f"[masking results]")
        print(f"  Total Gaussians: {num_gauss}")
        print(f"  Frames processed: {used}")
        print(f"  Min frames threshold: {min_frames} (ratio={args.min_frame_ratio})")
        print(f"  Missing masks: {missing_mask_count}")
        
        nonzero_mask = gaussian_frame_count > 0
        num_nonzero = int(nonzero_mask.sum().item())
        print(f"\n  ⭐ Gaussians with ANY appearance: {num_nonzero} ({100*num_nonzero/num_gauss:.3f}%)")
        
        if num_nonzero > 0:
            nonzero_counts = gaussian_frame_count[nonzero_mask]
            print(f"    Range: {nonzero_counts.min().item()}-{nonzero_counts.max().item()} frames")
            print(f"    Mean: {nonzero_counts.float().mean().item():.2f} frames")
            
            print(f"    Distribution:")
            for threshold in [1, 5, 10, 20, 30, 36, 40]:
                if threshold <= used:
                    count = (nonzero_counts >= threshold).sum().item()
                    if count > 0:
                        print(f"      >= {threshold:2d} frames: {count:6d} gaussians")
        
        print(f"\n  🎯 Selected for masking: {num_selected} ({100*num_selected/num_gauss:.3f}%)")
        
        if num_selected > 0:
            selected_counts = gaussian_frame_count[selected]
            print(f"    Range: {selected_counts.min().item()}-{selected_counts.max().item()} frames")
            print(f"    Mean: {selected_counts.float().mean().item():.2f} frames")

            # ========== 黑掉并保存 ==========
            print(f"\n{'='*60}")
            print(f"[mask] Blackening {num_selected} gaussians...")
            with torch.no_grad():
                model.features_dc.data[selected] = -255.0
                model.features_rest.data[selected] = 0.0
            print(f"✅ Model updated!")

            if args.output is not None:
                print(f"\n[save] Saving to {args.output}...")
                original = torch.load(args.checkpoint, map_location="cpu")
                original["pipeline"] = pipeline.state_dict()
                torch.save(original, args.output)
                print(f"✅ Checkpoint saved!")

            # ========== 渲染masked图像 ==========
            if args.render_output_root is not None:
                print(f"\n{'='*60}")
                print(f"[render] Rendering {min(used, args.max_frames)} masked images...")
                args.render_output_root.mkdir(parents=True, exist_ok=True)
                
                render_count = 0
                for idx in range(len(cameras_all)):
                    cam = cameras_all[idx : idx + 1]
                    meta = cam.metadata or {}
                    
                    cam_idx = None
                    for k in ["sensor_idxs", "sensor_idx", "cam_idx", "camera_idx", "camera_id"]:
                        if k in meta:
                            v = meta[k]
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                cam_idx = int(v.item())
                            else:
                                cam_idx = int(v) if v is not None else None
                            break
                    
                    if cam_idx is None or cam_idx != args.target_cam_idx:
                        continue
                    
                    if render_count >= args.max_frames:
                        break
                    
                    with torch.no_grad():
                        outputs = model.get_outputs_for_camera(cam)
                    
                    if "rgb" in outputs:
                        render_img = outputs["rgb"]
                    elif "image" in outputs:
                        render_img = outputs["image"]
                    elif "comp_rgb" in outputs:
                        render_img = outputs["comp_rgb"]
                    else:
                        print(f"  [error] No RGB output in keys: {list(outputs.keys())}")
                        render_count += 1
                        continue

                    render_img = render_img.detach()
                    
                    # ✅ 检查实际渲染分辨率
                    actual_H, actual_W = render_img.shape[-3:-1] if render_img.dim() == 4 else render_img.shape[:2]
                    print(f"  [debug] Camera reports: {W}x{H}")
                    print(f"  [debug] Actual render: {actual_W}x{actual_H}")
                    print(f"  [debug] Mask size: {mask.shape}")

                    if actual_H != H or actual_W != W:
                        print(f"  [WARNING] SIZE MISMATCH! Mask will be misaligned!")

                    out_name = f"cam{args.target_cam_idx}_frame{render_count:04d}_masked.png"
                    out_path = args.render_output_root / out_name
                    
                    try:
                        if render_img.dim() == 4:
                            img_tensor = render_img[0]
                        elif render_img.dim() == 3:
                            if render_img.shape[0] == 3:
                                img_tensor = render_img.permute(1, 2, 0)
                            else:
                                img_tensor = render_img
                        else:
                            raise ValueError(f"Unexpected shape: {render_img.shape}")
                        
                        img_tensor = img_tensor.detach()
                        img_np = (img_tensor.clamp(0, 1) * 255).byte().cpu().numpy()
                        Image.fromarray(img_np).save(out_path)
                        
                        if render_count < 3:
                            print(f"  ✅ Saved {out_path.name}")
                    
                    except Exception as e:
                        print(f"  ❌ Failed: {e}")
                    
                    render_count += 1
                
                print(f"✅ Rendered {render_count} images")
        
        else:
            print(f"\n⚠️  No gaussians selected")

    print(f"\n{'='*60}")
    print("✅ All done!")


if __name__ == "__main__":
    main()
