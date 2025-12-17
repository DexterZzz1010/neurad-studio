#!/usr/bin/env python3
"""
Test rasterization for cam_idx==0 frames + Gaussian masking voting logic.
Exactly preserves test_raster_cam0.py's structure, adds masking on top.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, Set, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import CameraType

# 按你要求用这个
from gsplat_original.rendering import rasterization, RollingShutterType


# ============================================================================
# 原test脚本的所有辅助函数 - 完全不动
# ============================================================================

def _override_config(cfg, ckpt_path: Path):
    # 让 eval_setup 正确加载 checkpoint
    cfg.load_dir = ckpt_path.parent
    name = ckpt_path.name
    try:
        step_str = name.split("-")[1].split(".")[0]
        cfg.load_step = int(step_str)
    except Exception:
        cfg.load_step = None

    # 强制禁用 dataloader/多进程
    if hasattr(cfg.pipeline, "datamanager"):
        cfg.pipeline.datamanager.max_thread_workers = 0
        if hasattr(cfg.pipeline.datamanager, "num_processes"):
            cfg.pipeline.datamanager.num_processes = 0

    # 保险：别搞 ray tracing
    if hasattr(cfg.pipeline, "model") and hasattr(cfg.pipeline.model, "use_ray_tracing"):
        cfg.pipeline.model.use_ray_tracing = False

    return cfg


def _camera_to_viewmat(camera, device: torch.device) -> torch.Tensor:
    c2w = camera.camera_to_worlds.to(device)
    if c2w.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1], dtype=c2w.dtype, device=device).view(1, 1, 4)
        c2w = torch.cat([c2w, bottom], dim=-2)  # [1,4,4]
    elif c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Unexpected camera_to_worlds shape: {tuple(c2w.shape)}")
    return torch.linalg.inv(c2w)  # world->cam


def _camera_to_K(camera, device: torch.device) -> torch.Tensor:
    fx = camera.fx.to(device).view(-1)[0]
    fy = camera.fy.to(device).view(-1)[0]
    cx = camera.cx.to(device).view(-1)[0]
    cy = camera.cy.to(device).view(-1)[0]
    K = torch.zeros((1, 3, 3), dtype=fx.dtype, device=device)
    K[0, 0, 0] = fx
    K[0, 1, 1] = fy
    K[0, 0, 2] = cx
    K[0, 1, 2] = cy
    K[0, 2, 2] = 1.0
    return K


def _build_distortion_kwargs(camera, device) -> Dict[str, torch.Tensor]:
    if camera.distortion_params is None:
        return {}
    params = camera.distortion_params.to(device)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    if not torch.any(params != 0):
        return {}

    # radial: pad to 6
    radial = torch.zeros(params.shape[:-1] + (6,), device=device, dtype=params.dtype)
    radial[..., : min(4, params.shape[-1])] = params[..., : min(4, params.shape[-1])]
    kwargs: Dict[str, torch.Tensor] = {"radial_coeffs": radial}

    if params.shape[-1] >= 6:
        tangential = params[..., 4:6]
        if torch.any(tangential != 0):
            kwargs["tangential_coeffs"] = tangential
    return kwargs


def _extract_gaussians_for_raster(model, device: torch.device):
    # 这块只做一次，避免每帧重复
    if not hasattr(model, "means"):
        raise AttributeError("model missing `means`")

    means = model.means.contiguous().to(device)

    if hasattr(model, "quats"):
        quats = model.quats.contiguous().to(device)
    elif hasattr(model, "rotations"):
        quats = model.rotations.contiguous().to(device)
    else:
        raise AttributeError("model missing quats/rotations")

    if not hasattr(model, "scales"):
        raise AttributeError("model missing `scales` (expected log-scales)")
    scales = torch.exp(model.scales).contiguous().to(device)

    if not hasattr(model, "opacities"):
        raise AttributeError("model missing `opacities`")
    opacities = torch.sigmoid(model.opacities).squeeze(-1).contiguous().to(device)

    # SH colors: [N,K,3]
    if not hasattr(model, "features_dc") or not hasattr(model, "features_rest"):
        raise AttributeError("model missing SH features (features_dc/features_rest)")

    N = model.features_dc.shape[0]
    sh_degree = int(getattr(model.config, "sh_degree", 3))
    K = (sh_degree + 1) ** 2
    exp_dim = (K - 1) * 3
    act_dim = model.features_rest.shape[1]
    if act_dim != exp_dim:
        # 自动兜底（宁可跑通）
        K = act_dim // 3 + 1
        sh_degree = int(np.sqrt(K)) - 1

    colors_rest = model.features_rest.view(N, K - 1, 3)
    colors_sh = torch.cat([model.features_dc[:, None, :], colors_rest], dim=1).contiguous().to(device)

    return means, quats, scales, opacities, colors_sh, sh_degree


def _debug_cuda(tag: str, sync: bool):
    if torch.cuda.is_available():
        if sync:
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"[cuda] {tag}: used={used/1e9:.2f}GB free={free/1e9:.2f}GB total={total/1e9:.2f}GB")


# ============================================================================
# Masking相关的新增函数 - 独立添加，不影响原有逻辑
# ============================================================================

def _load_mask(mask_path: Path, size: tuple) -> torch.Tensor:
    """Load and resize mask to target size."""
    img = Image.open(mask_path).convert("L").resize(size, resample=Image.NEAREST)
    arr = np.array(img)
    return torch.from_numpy(arr >= 1)


def _tiles_from_mask(mask: torch.Tensor, tile_size: int, tile_width: int) -> torch.Tensor:
    """Convert mask to a flat list of tile indices that have any positive pixels."""
    H, W = mask.shape
    tile_h = (H + tile_size - 1) // tile_size
    tile_w = tile_width  # already provided by rasterization meta
    pad_h = max(0, tile_h * tile_size - H)
    pad_w = max(0, tile_w * tile_size - W)

    # Pad to full tiles then reduce; keep it simple/CPU-friendly.
    m = mask
    if pad_h > 0 or pad_w > 0:
        m = F.pad(m, (0, pad_w, 0, pad_h))
    m = m.view(tile_h, tile_size, tile_w, tile_size)
    tile_hit = m.any(dim=3).any(dim=1)  # [tile_h, tile_w]
    tiles = tile_hit.flatten().nonzero(as_tuple=False).squeeze(1)  # [N_tiles]
    return tiles


def _vote_gaussians_from_mask(
    mask: torch.Tensor,
    info: dict,
    opacities: torch.Tensor,
    visible_ids: torch.Tensor,
    min_alpha: float,
    num_gauss_total: int,
    verbose: bool = False,
) -> Set[int]:
    """
    给定mask和rasterization info，投票选出mask区域内的高斯。
    
    Args:
        mask: [H, W] bool tensor, mask区域
        info: rasterization返回的info字典
        opacities: [num_gauss_total] 全局opacity数组
        visible_ids: [N_visible] 可见高斯的全局ID
        min_alpha: 最小alpha阈值
        num_gauss_total: 全局高斯总数
        verbose: 是否打印调试信息
    
    Returns:
        Set[int]: 全局高斯ID的集合
    """
    isect_offsets = info["isect_offsets"].reshape(-1)
    flatten_ids = info["flatten_ids"]
    tile_size = int(info["tile_size"].item() if isinstance(info["tile_size"], torch.Tensor) else info["tile_size"])
    tile_width = int(info["tile_width"].item() if isinstance(info["tile_width"], torch.Tensor) else info["tile_width"])

    if verbose:
        print(f"    [vote_debug] visible_ids.shape={visible_ids.shape}")
        print(f"    [vote_debug] flatten_ids: shape={flatten_ids.shape}, min={flatten_ids.min().item()}, max={flatten_ids.max().item()}")
        print(f"    [vote_debug] num_gauss_total={num_gauss_total}")
        print(f"    [vote_debug] tile_size={tile_size}, tile_width={tile_width}")

    # ✅ mask → tile 列表（大幅减少循环量）
    tiles = _tiles_from_mask(mask, tile_size, tile_width)
    if tiles.numel() == 0:
        if verbose:
            print("    [vote_debug] No tiles hit by mask")
        return set()

    # ✅ 可见集合做成 bool 索引，避免分配全局到局部映射的长 tensor
    device = visible_ids.device
    vis_flag = torch.zeros((num_gauss_total,), dtype=torch.bool, device=device)
    vis_flag[visible_ids] = True

    frame_gaussians: Set[int] = set()
    total_candidates = 0
    total_visible_candidates = 0

    # tile 数通常远少于像素，用 Python 循环够快
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
        print(f"    [vote_debug] Total candidate entries: {total_candidates}")
        print(f"    [vote_debug] Visible candidates: {total_visible_candidates}")
        print(f"    [vote_debug] Unique gaussians voted: {len(frame_gaussians)}")
        if len(frame_gaussians) > 0:
            sample = list(frame_gaussians)[:5]
            print(f"    [vote_debug] Sample global IDs: {sample}")

    return frame_gaussians


# ============================================================================
# Main - 完全保留test逻辑，添加masking分支
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    
    # 原test参数
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=10, help="最多测试多少帧 cam_idx==0")
    ap.add_argument("--sync-cuda", action="store_true", help="每个关键点 synchronize，定位卡点")
    ap.add_argument("--packed", action="store_true", help="用 packed=True（强烈建议）")
    ap.add_argument("--radius-clip", type=float, default=None, help="覆盖 radius_clip_pix（可用于加速）")
    ap.add_argument("--render-mode", type=str, default="RGB")
    
    # 新增masking参数
    ap.add_argument("--enable-masking", action="store_true", help="启用masking逻辑")
    ap.add_argument("--mask-root", type=Path, default=None, help="Mask根目录")
    ap.add_argument("--mask-template", default="car_mask_001/id1_{frame_idx:02d}_mask.png")  # ✅ 修复：加上右括号
    ap.add_argument("--target-cam-idx", type=int, default=0)
    ap.add_argument("--min-alpha", type=float, default=0.01)
    ap.add_argument("--min-frame-ratio", type=float, default=0.9)
    ap.add_argument("--output", type=Path, default=None, help="如果指定，保存masked checkpoint")
    
    args = ap.parse_args()
    
    # ✅✅✅ 立即在解析后修复template（如果bash传错了） ✅✅✅
    if args.enable_masking:
        # 检测并修复常见的bash传递错误
        if '_mask.png' in args.mask_template and '02d_mask.png' in args.mask_template:
            # Bash吃掉了右括号
            print(f"[warning] Detected malformed template: {args.mask_template!r}")
            args.mask_template = args.mask_template.replace(':02d_mask.png', ':02d}_mask.png')
            print(f"[warning] Auto-fixed to: {args.mask_template!r}")
    device = torch.device(args.device)

    print("[setup] loading pipeline...")
    config, pipeline, _, step = eval_setup(
        args.config,
        test_mode="inference",
        update_config_callback=lambda cfg: _override_config(cfg, args.checkpoint),
    )
    pipeline.to(device)
    model = pipeline.model
    model.eval()

    print("✅ Done loading checkpoint from", args.checkpoint)
    print("[info] Device:", device)
    print("[info] Model:", type(model).__name__)

    # dataset cameras
    if args.split == "eval":
        dataset = pipeline.datamanager.eval_dataset
    else:
        dataset = pipeline.datamanager.train_dataset
    cameras_all = dataset.cameras.to(device)
    print("[info] Dataset:", len(cameras_all), "cameras")

    # gaussians
    print("[setup] extracting gaussians once...")
    _debug_cuda("before_extract", args.sync_cuda)
    means, quats, scales, opacities, colors_sh, sh_degree = _extract_gaussians_for_raster(model, device)
    num_gauss = means.shape[0]
    print("[info] Total Gaussians:", num_gauss)
    _debug_cuda("after_extract", args.sync_cuda)

    # Masking初始化（仅当启用时）
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
        print(f"[masking] Template: {args.mask_template}")

    # iterate cam_idx==target
    used = 0
    for idx in range(len(cameras_all)):
        cam = cameras_all[idx : idx + 1]
        meta = cam.metadata or {}

        # 提取 cam_idx（兼容不同字段名）
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
        print(f"\n[frame] idx={idx} cam_idx={args.target_cam_idx}  size={W}x{H}")

        # camera model
        camera_model = getattr(model.config, "camera_model", "pinhole")
        if hasattr(cam, "camera_type"):
            ct = int(cam.camera_type.view(-1)[0].item())
            if ct == CameraType.FISHEYE.value:
                camera_model = "fisheye"

        viewmat = _camera_to_viewmat(cam, device)
        Kmat = _camera_to_K(cam, device)
        raster_kwargs = _build_distortion_kwargs(cam, device)

        radius_clip = getattr(model.config, "radius_clip_pix", 0.0)
        if args.radius_clip is not None:
            radius_clip = float(args.radius_clip)

        packed = False

        # debug prints
        print(f"[debug] packed={packed} sh_degree={sh_degree} camera_model={camera_model} radius_clip={radius_clip}")
        print(f"[debug] distortion keys={list(raster_kwargs.keys())}")

        _debug_cuda("before_raster", args.sync_cuda)
        if args.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        print("[debug] before rasterization")

        with torch.no_grad():
            render, alpha, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_sh,
                viewmats=viewmat,
                Ks=Kmat,
                width=W,
                height=H,
                near_plane=0.5,
                far_plane=1e10,
                radius_clip=radius_clip,
                eps2d=0.3,
                sh_degree=sh_degree,
                packed=packed,
                tile_size=16,
                backgrounds=None,
                render_mode=args.render_mode,
                sparse_grad=False,
                absgrad=getattr(model.config, "use_absgrad", False),
                rasterize_mode=getattr(model.config, "rasterize_mode", "classic"),
                channel_chunk=128,
                distributed=False,
                camera_model=camera_model,
                segmented=False,
                covars=None,
                with_ut=getattr(model.config, "with_ut", False),
                with_eval3d=getattr(model.config, "with_eval3d", False),
                radial_coeffs=raster_kwargs.get("radial_coeffs"),
                tangential_coeffs=raster_kwargs.get("tangential_coeffs"),
                thin_prism_coeffs=raster_kwargs.get("thin_prism_coeffs"),
                rolling_shutter=RollingShutterType.GLOBAL,
                viewmats_rs=None,
            )

        if args.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        print("[debug] after rasterization  time={:.3f}s".format(dt))
        _debug_cuda("after_raster", args.sync_cuda)

        # 输出 render/alpha/info 形状与 key
        try:
            rshape = tuple(render.shape)
        except Exception:
            rshape = str(type(render))
        try:
            ashape = tuple(alpha.shape)
        except Exception:
            ashape = str(type(alpha))
        print(f"[debug] render.shape={rshape} alpha.shape={ashape}")

        if isinstance(info, dict):
            keys = list(info.keys())
            print(f"[debug] info keys ({len(keys)}): {keys[:40]}{'...' if len(keys)>40 else ''}")
            need = ["means2d", "conics", "isect_offsets", "flatten_ids", "tile_width", "tile_size", "gaussian_ids"]
            present = {k: (k in info) for k in need}
            print("[debug] info has:", present)
        else:
            print("[debug] info type:", type(info))

# ========== 新增：Masking逻辑（完全独立的分支） ==========
        if args.enable_masking:
            print(f"  [masking] begin masking")
            # 构造format context
            fmt_ctx = {}
            for k, v in meta.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    fmt_ctx[k] = v.item()
                elif not isinstance(v, torch.Tensor):
                    fmt_ctx[k] = v

            fmt_ctx["frame_idx"] = cam_frame_counter
            fmt_ctx["cam_idx"] = cam_idx

            # ✅✅✅ 直接打印，看看到底发生了什么 ✅✅✅
            if used < 3:
                print(f"  [debug] Format context: {fmt_ctx}")
                print(f"  [debug] Template: {args.mask_template!r}")  # !r 显示原始字符串
            
            try:
                mask_name = args.mask_template.format(**fmt_ctx)
                if used < 3:
                    print(f"  [debug] Formatted mask_name: {mask_name!r}")
            except (KeyError, ValueError) as e:
                print(f"  [skip] Cannot format mask template: {e}")
                print(f"  [skip]   Template: {args.mask_template!r}")
                print(f"  [skip]   Context: {fmt_ctx}")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

            mask_path = args.mask_root / mask_name
            
            if used < 3:
                print(f"  [debug] Full mask path: {mask_path}")
                print(f"  [debug] Path exists: {mask_path.exists()}")
                
                # # ✅ 列出mask目录，看看实际有什么文件
                # mask_dir = args.mask_root / "car_mask_001"
                # if mask_dir.exists():
                #     files = sorted(mask_dir.glob("*.png"))[:10]
                #     print(f"  [debug] Sample files in {mask_dir}:")
                #     for f in files:
                #         print(f"  [debug]   - {f.name}")
                # else:
                #     print(f"  [debug] Mask directory does not exist: {mask_dir}")

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
                print(f"  [skip] Empty mask: {mask_path.name}")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

            print(f"  [masking] ✅ Loaded: {mask_path.name}, pixels={mask_pixel_count}")

#

            # ✅ 获取visible_ids（移到前面，先检查）
            visible_ids = None
            # 优先使用 rasterization 返回的可见 ID
            if "gaussian_ids" in info and info["gaussian_ids"] is not None:
                gids = info["gaussian_ids"]
                if isinstance(gids, torch.Tensor) and gids.ndim == 2:
                    gids = gids[0]
                visible_ids = gids if isinstance(gids, torch.Tensor) else torch.as_tensor(gids, device=device)
            # 次优：根据 tiles_per_gauss 判断可见（比全量更小）
            elif "tiles_per_gauss" in info and info["tiles_per_gauss"] is not None:
                tpg = info["tiles_per_gauss"]
                if isinstance(tpg, torch.Tensor):
                    tpg = tpg.reshape(-1, tpg.shape[-1])  # [BC, N]
                    vis_mask = tpg.any(dim=0)
                    visible_ids = vis_mask.nonzero(as_tuple=False).squeeze(1)

            if visible_ids is None:
                print(f"  [ERROR] gaussian_ids not found in info! Using fallback (BAD!)")
                print(f"  [ERROR] This means rasterization didn't return visible gaussian IDs")
                visible_ids = torch.arange(num_gauss, device=device)
            
            # ✅ 打印可见高斯信息（每一帧）
            num_visible = len(visible_ids)
            print(f"  [masking] Visible gaussians: {num_visible} / {num_gauss} ({100*num_visible/num_gauss:.2f}%)")
            
            if num_visible == 0:
                print(f"  [ERROR] No visible gaussians! Skipping this frame.")
                cam_frame_counter += 1
                used += 1
                if used >= args.max_frames:
                    break
                continue

#

            # ✅ 详细调试信息（前3帧或者有问题的帧）
            if used < 3:
                print(f"  [debug] === Frame {used} detailed info ===")
                print(f"  [debug] info keys: {list(info.keys())}")
                
                if "gaussian_ids" in info:
                    gids = info["gaussian_ids"]
                    if gids is not None:
                        if isinstance(gids, torch.Tensor):
                            print(f"  [debug] gaussian_ids: shape={gids.shape}, dtype={gids.dtype}")
                            print(f"  [debug] gaussian_ids: min={gids.min().item()}, max={gids.max().item()}")
                            print(f"  [debug] gaussian_ids[:10]: {gids[:10].tolist()}")
                        else:
                            print(f"  [debug] gaussian_ids: type={type(gids)}")
                    else:
                        print(f"  [debug] gaussian_ids is None!")
                else:
                    print(f"  [debug] 'gaussian_ids' not in info!")
                
                print(f"  [debug] means2d: shape={info['means2d'].shape}, dtype={info['means2d'].dtype}")
                print(f"  [debug] conics: shape={info['conics'].shape}, dtype={info['conics'].dtype}")
                print(f"  [debug] flatten_ids: shape={info['flatten_ids'].shape}, dtype={info['flatten_ids'].dtype}")
                print(f"  [debug] flatten_ids: min={info['flatten_ids'].min().item()}, max={info['flatten_ids'].max().item()}")
                print(f"  [debug] flatten_ids[:20]: {info['flatten_ids'][:20].tolist()}")
                print(f"  [debug] isect_offsets: shape={info['isect_offsets'].shape}")
                print(f"  [debug] tile_size: {info['tile_size']}")
                print(f"  [debug] tile_width: {info['tile_width']}")
                
                # ✅ 检查visible_ids的有效性
                print(f"  [debug] visible_ids: shape={visible_ids.shape}, dtype={visible_ids.dtype}")
                print(f"  [debug] visible_ids: min={visible_ids.min().item()}, max={visible_ids.max().item()}")
                print(f"  [debug] visible_ids[:10]: {visible_ids[:10].tolist()}")
                
                # ✅ 检查flatten_ids的取模结果
                sample_flatten = info['flatten_ids'][:20]
                sample_modulo = sample_flatten % num_gauss
                print(f"  [debug] Sample flatten_ids % num_gauss: {sample_modulo.tolist()}")
                
                # ✅ 检查这些ID是否在visible_ids中
                in_visible = torch.isin(sample_modulo, visible_ids)
                print(f"  [debug] Sample IDs in visible_ids: {in_visible.tolist()} ({in_visible.sum().item()}/{len(in_visible)} found)")
#
            # ✅ Voting with debug info
            frame_gauss_set = _vote_gaussians_from_mask(
                mask=mask,
                info=info,
                opacities=opacities,
                visible_ids=visible_ids,
                min_alpha=args.min_alpha,
                num_gauss_total=num_gauss,
                verbose=(used < 3),  # 仅前3帧详细调试
            )

            # ✅ 输出投票结果（每一帧）
            if len(frame_gauss_set) > 0:
                gauss_tensor = torch.tensor(list(frame_gauss_set), dtype=torch.long, device=device)
                gaussian_frame_count[gauss_tensor] += 1
                print(f"  [masking] ✅ Voted {len(frame_gauss_set)} gaussians")
                
                if used < 3:
                    sample_ids = sorted(list(frame_gauss_set))[:10]
                    print(f"  [masking] Sample global IDs: {sample_ids}")
            else:
                print(f"  [masking] ⚠️  WARNING: No gaussians voted for this frame!")
                if used < 3:
                    print(f"  [masking] This is FRAME {used} - should investigate why!")

            cam_frame_counter += 1

        # ========== 原test逻辑继续 ==========
        used += 1
        if used >= args.max_frames:
            break

    print(f"\n[done] tested frames(cam_idx={args.target_cam_idx}): {used}")

    # ========== Masking输出（如果启用） ==========
    if args.enable_masking and gaussian_frame_count is not None:
        min_frames = int(used * args.min_frame_ratio)
        selected = gaussian_frame_count >= min_frames
        num_selected = int(selected.sum().item())

        print(f"\n[masking results]")
        print(f"  Total Gaussians: {num_gauss}")
        print(f"  Frames processed: {used}")
        print(f"  Min frames threshold: {min_frames} (ratio={args.min_frame_ratio})")
        print(f"  Selected: {num_selected} ({100*num_selected/num_gauss:.3f}%)")
        print(f"  Missing masks: {missing_mask_count}")
        
        # ✅ 统计所有非零计数（这个最重要！）
        nonzero_mask = gaussian_frame_count > 0
        num_nonzero = int(nonzero_mask.sum().item())
        print(f"\n  ⭐ Gaussians with ANY appearance: {num_nonzero} ({100*num_nonzero/num_gauss:.3f}%)")
        
        if num_nonzero > 0:
            nonzero_counts = gaussian_frame_count[nonzero_mask]
            print(f"    Appearance range: {nonzero_counts.min().item()}-{nonzero_counts.max().item()} frames")
            print(f"    Mean appearances: {nonzero_counts.float().mean().item():.2f} frames")
            
            # ✅ 分布直方图
            print(f"    Distribution:")
            for threshold in [1, 5, 10, 20, 30, 36, 40]:
                count = (nonzero_counts >= threshold).sum().item()
                if count > 0:
                    print(f"      >= {threshold} frames: {count} gaussians")
        else:
            print(f"  ❌ CRITICAL: No gaussians were voted in ANY frame!")
            print(f"  This means the voting logic is completely broken.")
        
        # ✅ 统计分布
        if num_selected > 0:
            selected_counts = gaussian_frame_count[selected]
            print(f"\n  Selected gaussians stats:")
            print(f"    Min: {selected_counts.min().item()} frames")
            print(f"    Max: {selected_counts.max().item()} frames")
            print(f"    Mean: {selected_counts.float().mean().item():.2f} frames")

        if args.output:
            if num_selected > 0:
                print(f"\n[save] Blackening {num_selected} gaussians...")
                with torch.no_grad():
                    model.features_dc[selected] = 0.0
                    model.features_rest[selected] = 0.0

                print(f"[save] Saving to {args.output}...")
                original = torch.load(args.checkpoint, map_location="cpu")
                original["pipeline"] = pipeline.state_dict()
                torch.save(original, args.output)
                print(f"✅ Masked checkpoint saved!")
            else:
                print(f"\n[save] ⚠️  No gaussians selected, skipping checkpoint save")
                if num_nonzero > 0:
                    print(f"[save] Hint: {num_nonzero} gaussians appeared in some frames,")
                    print(f"[save]       but none appeared in >= {min_frames} frames.")
                    print(f"[save]       Try lowering --min-frame-ratio (current: {args.min_frame_ratio})")


if __name__ == "__main__":
    main()
