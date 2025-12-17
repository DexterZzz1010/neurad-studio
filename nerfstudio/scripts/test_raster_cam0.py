#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import CameraType

# 按你要求用这个
from gsplat_original.rendering import rasterization, RollingShutterType


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=10, help="最多测试多少帧 cam_idx==0")
    ap.add_argument("--sync-cuda", action="store_true", help="每个关键点 synchronize，定位卡点")
    ap.add_argument("--packed", action="store_true", help="用 packed=True（强烈建议）")
    ap.add_argument("--radius-clip", type=float, default=None, help="覆盖 radius_clip_pix（可用于加速）")
    ap.add_argument("--render-mode", type=str, default="RGB")
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
    print("[info] Total Gaussians:", means.shape[0])
    _debug_cuda("after_extract", args.sync_cuda)

    # iterate cam_idx==0
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

        if cam_idx is None or cam_idx != 0:
            continue

        W, H = int(cam.width.item()), int(cam.height.item())
        print(f"\n[frame] idx={idx} cam_idx=0  size={W}x{H}")

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
                radius_clip=3.0,
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

        # 输出 render/alpha/info 形状与 key，确认是否拿到了你需要的 info
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
            # 重点检查你后面 voting 需要的字段
            need = ["means2d", "conics", "isect_offsets", "flatten_ids", "tile_width", "tile_size", "gaussian_ids"]
            present = {k: (k in info) for k in need}
            print("[debug] info has:", present)
        else:
            print("[debug] info type:", type(info))

        used += 1
        if used >= args.max_frames:
            break

    print(f"\n[done] tested frames(cam_idx==0): {used}")


if __name__ == "__main__":
    main()
