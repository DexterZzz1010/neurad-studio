#!/usr/bin/env python3
"""
EXACT copy of test script logic + mask voting.
Critical: Filter BEFORE rasterization, not after!
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import CameraType

from gsplat_original.rendering import rasterization, RollingShutterType


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


def _camera_to_viewmat(camera, device: torch.device) -> torch.Tensor:
    c2w = camera.camera_to_worlds.to(device)
    if c2w.shape[-2:] == (3, 4):
        bottom = torch.tensor([0, 0, 0, 1], dtype=c2w.dtype, device=device).view(1, 1, 4)
        c2w = torch.cat([c2w, bottom], dim=-2)
    return torch.linalg.inv(c2w)


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

    radial = torch.zeros(params.shape[:-1] + (6,), device=device, dtype=params.dtype)
    radial[..., : min(4, params.shape[-1])] = params[..., : min(4, params.shape[-1])]
    kwargs: Dict[str, torch.Tensor] = {"radial_coeffs": radial}

    if params.shape[-1] >= 6:
        tangential = params[..., 4:6]
        if torch.any(tangential != 0):
            kwargs["tangential_coeffs"] = tangential
    return kwargs


def _extract_gaussians_for_raster(model, device: torch.device):
    if not hasattr(model, "means"):
        raise AttributeError("model missing `means`")

    means = model.means.contiguous().to(device)

    if hasattr(model, "quats"):
        quats = model.quats.contiguous().to(device)
    elif hasattr(model, "rotations"):
        quats = model.rotations.contiguous().to(device)
    else:
        raise AttributeError("model missing quats/rotations")

    scales = torch.exp(model.scales).contiguous().to(device)
    opacities = torch.sigmoid(model.opacities).squeeze(-1).contiguous().to(device)

    N = model.features_dc.shape[0]
    sh_degree = int(getattr(model.config, "sh_degree", 3))
    K = (sh_degree + 1) ** 2
    exp_dim = (K - 1) * 3
    act_dim = model.features_rest.shape[1]
    if act_dim != exp_dim:
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


def _load_mask(mask_path: Path, size: tuple) -> torch.Tensor:
    img = Image.open(mask_path).convert("L").resize(size, resample=Image.NEAREST)
    arr = np.array(img)
    return torch.from_numpy(arr >= 1)


def collect_gaussians_in_pixel(
    x: int,
    y: int,
    tile_size: int,
    tile_width: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    min_alpha: float,
    num_gaussians: int,
) -> Set[int]:
    """Collect gaussians at pixel."""
    tile_x = x // tile_size
    tile_y = y // tile_size
    tile_linear = tile_y * tile_width + tile_x

    start = isect_offsets[tile_linear]
    end = flatten_ids.numel() if tile_linear + 1 == isect_offsets.numel() else isect_offsets[tile_linear + 1]

    if end <= start:
        return set()

    candidate_ids = flatten_ids[start:end] % num_gaussians

    valid_gaussians = set()
    alpha_accum = 0.0

    for gid in candidate_ids.tolist():
        mean = means2d[gid]
        conic = conics[gid]
        dx = x - mean[0]
        dy = y - mean[1]
        exponent = -0.5 * (conic[0] * dx * dx + 2 * conic[1] * dx * dy + conic[2] * dy * dy)
        alpha = opacities[gid] * torch.exp(exponent)

        if alpha > min_alpha:
            valid_gaussians.add(gid)

        alpha_accum += alpha * (1.0 - alpha_accum)
        if alpha_accum > 0.99:
            break

    return valid_gaussians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("mask_root", type=Path)
    ap.add_argument("--mask-template", default="car_mask_001/id1_{frame_idx:02d}_mask.png")
    ap.add_argument("--target-cam-idx", type=int, default=0)
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--min-alpha", type=float, default=0.01)
    ap.add_argument("--min-frame-ratio", type=float, default=0.9)
    ap.add_argument("--output", type=Path, default=Path("masked.ckpt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--radius-clip", type=float, default=3.0)  # ✅ 默认3.0
    ap.add_argument("--render-mode", type=str, default="RGB")
    ap.add_argument("--sync-cuda", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

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

    # Dataset
    if args.split == "eval":
        dataset = pipeline.datamanager.eval_dataset
    else:
        dataset = pipeline.datamanager.train_dataset
    cameras_all = dataset.cameras.to(device)
    print("[info] Dataset:", len(cameras_all), "cameras")

    # ✅ Extract gaussians ONCE (as test script)
    print("[setup] extracting gaussians once...")
    _debug_cuda("before_extract", args.sync_cuda)
    means, quats, scales, opacities, colors_sh, sh_degree = _extract_gaussians_for_raster(model, device)
    num_gauss = means.shape[0]
    print("[info] Total Gaussians:", num_gauss)
    _debug_cuda("after_extract", args.sync_cuda)

    # Counter
    gaussian_frame_count = torch.zeros(num_gauss, dtype=torch.int32, device=device)

    # Stats
    used = 0
    missing_mask = 0
    cam0_counter = 0

    print(f"\n[processing] Target cam_idx={args.target_cam_idx}...")

    # ✅ EXACT loop as test script
    for idx in range(len(cameras_all)):
        cam = cameras_all[idx : idx + 1]
        meta = cam.metadata or {}

        # ✅ Extract cam_idx (as test)
        cam_idx = None
        for k in ["sensor_idxs", "sensor_idx", "cam_idx", "camera_idx", "camera_id"]:
            if k in meta:
                v = meta[k]
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    cam_idx = int(v.item())
                else:
                    cam_idx = int(v) if v is not None else None
                break

        # ✅ CRITICAL: Filter BEFORE rasterization (as test)
        if cam_idx is None or cam_idx != args.target_cam_idx:
            continue  # ✅ Skip immediately, no rasterization!

        # ✅ Now we only process cam_idx=target frames
        W, H = int(cam.width.item()), int(cam.height.item())
        
        if used < 3:
            print(f"\n[frame] idx={idx} cam_idx={cam_idx}  size={W}x{H}")

        # Camera model (as test)
        camera_model = getattr(model.config, "camera_model", "pinhole")
        if hasattr(cam, "camera_type"):
            ct = int(cam.camera_type.view(-1)[0].item())
            if ct == CameraType.FISHEYE.value:
                camera_model = "fisheye"

        viewmat = _camera_to_viewmat(cam, device)
        Kmat = _camera_to_K(cam, device)
        raster_kwargs = _build_distortion_kwargs(cam, device)

        radius_clip = args.radius_clip  # ✅ Use command line arg

        packed = False

        # Debug (as test)
        if used < 3:
            print(f"[debug] packed={packed} sh_degree={sh_degree} camera_model={camera_model} radius_clip={radius_clip}")
            print(f"[debug] distortion keys={list(raster_kwargs.keys())}")
            _debug_cuda("before_raster", args.sync_cuda)

        if args.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t0 = time.time()
        if used < 3:
            print("[debug] before rasterization")

        # ✅ EXACT rasterization call (as test)
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
        
        if used < 3:
            print(f"[debug] after rasterization  time={dt:.3f}s")
            _debug_cuda("after_raster", args.sync_cuda)

        # ============================================================
        # Load mask
        # ============================================================

        fmt_ctx = {}
        for k, v in meta.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                fmt_ctx[k] = v.item()
            elif not isinstance(v, torch.Tensor):
                fmt_ctx[k] = v

        fmt_ctx["frame_idx"] = cam0_counter
        fmt_ctx["cam_idx"] = cam_idx

        try:
            mask_name = args.mask_template.format(**fmt_ctx)
        except (KeyError, ValueError):
            cam0_counter += 1
            continue

        mask_path = args.mask_root / mask_name

        if not mask_path.exists():
            if args.verbose and missing_mask < 5:
                print(f"[skip] Missing: {mask_path}")
            missing_mask += 1
            cam0_counter += 1
            continue

        mask = _load_mask(mask_path, size=(W, H)).to(device)
        if mask.sum().item() == 0:
            cam0_counter += 1
            continue

        if used < 3:
            print(f"  Mask: {mask_path.name}, pixels={int(mask.sum().item())}")

        # ============================================================
        # Voting
        # ============================================================

        means2d = info["means2d"]
        conics = info["conics"]

        if means2d.ndim == 3:
            means2d = means2d[0]
            conics = conics[0]

        isect_offsets = info["isect_offsets"].reshape(-1)
        flatten_ids = info["flatten_ids"]
        tile_size = int(info["tile_size"].item() if isinstance(info["tile_size"], torch.Tensor) else info["tile_size"])
        tile_width = int(info["tile_width"].item() if isinstance(info["tile_width"], torch.Tensor) else info["tile_width"])

        visible_ids = None
        if "gaussian_ids" in info and info["gaussian_ids"] is not None:
            gids = info["gaussian_ids"]
            if isinstance(gids, torch.Tensor) and gids.ndim == 2:
                gids = gids[0]
            visible_ids = gids if isinstance(gids, torch.Tensor) else torch.as_tensor(gids, device=device)

        if visible_ids is None:
            visible_ids = torch.arange(num_gauss, device=device)

        visible_opacities = opacities[visible_ids]

        frame_gaussians: Set[int] = set()

        ys, xs = torch.where(mask)

        for y, x in zip(ys.tolist(), xs.tolist()):
            pixel_gaussians = collect_gaussians_in_pixel(
                x=int(x),
                y=int(y),
                tile_size=tile_size,
                tile_width=tile_width,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                means2d=means2d,
                conics=conics,
                opacities=visible_opacities,
                min_alpha=args.min_alpha,
                num_gaussians=len(visible_ids),
            )
            frame_gaussians.update(pixel_gaussians)

        if frame_gaussians:
            global_frame_gaussians = set(visible_ids[list(frame_gaussians)].tolist())
            gauss_tensor = torch.tensor(list(global_frame_gaussians), dtype=torch.long, device=device)
            gaussian_frame_count[gauss_tensor] += 1

        used += 1
        cam0_counter += 1

    print(f"\n[done] tested frames(cam_idx={args.target_cam_idx}): {used}")

    # ============================================================
    # Select
    # ============================================================

    min_frames = int(used * args.min_frame_ratio)
    selected = gaussian_frame_count >= min_frames
    num_selected = int(selected.sum().item())

    print(f"\n[results] Gaussians:")
    print(f"  Total: {num_gauss}")
    print(f"  Selected: {num_selected} ({100*num_selected/num_gauss:.3f}%)")

    # Save
    print(f"\n[save] Blackening {num_selected} gaussians...")
    with torch.no_grad():
        model.features_dc[selected] = 0.0
        model.features_rest[selected] = 0.0

    print(f"[save] Saving to {args.output}...")
    original = torch.load(args.checkpoint, map_location="cpu")
    original["pipeline"] = pipeline.state_dict()
    torch.save(original, args.output)

    print(f"\n✅ Success!")


if __name__ == "__main__":
    main()