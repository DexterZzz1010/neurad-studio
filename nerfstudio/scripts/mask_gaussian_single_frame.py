#!/usr/bin/env python3
"""
测试mask选高斯准确性：每帧独立mask→渲染→复原
修复版：正确处理跳帧数据集
"""
import argparse
import os
from pathlib import Path
import re
import torch
import numpy as np
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup

def _override_config(cfg, ckpt_path: Path):
    cfg.load_dir = ckpt_path.parent
    if hasattr(cfg.pipeline, "datamanager"):
        cfg.pipeline.datamanager.max_thread_workers = 0
    if hasattr(cfg.pipeline, "model") and hasattr(cfg.pipeline.model, "use_ray_tracing"):
        cfg.pipeline.model.use_ray_tracing = False
    return cfg

def extract_real_frame_idx(
    meta: dict, fallback: int, image_filename: str = None, image_name: str = None
) -> int:
    """
    从metadata中提取真实的帧号
    尝试多个可能的key
    """
    # 尝试常见的key
    for k in ["frame_num", "frame_idx", "frame_id", "image_idx", "timestamp_idx"]:
        if k in meta:
            v = meta[k]
            if isinstance(v, torch.Tensor):
                return int(v.item())
            else:
                return int(v)
    
    # 尝试从filepath解析
    if 'filepath' in meta:
        filepath = meta['filepath']
        if isinstance(filepath, (list, tuple)):
            filepath = filepath[0]
        
        # 匹配数字.png 或 /数字.png
        match = re.search(r'[/\\]?(\d+)\.png', str(filepath))
        if match:
            return int(match.group(1))
    
    # 尝试从image_name解析
    if 'image_name' in meta:
        name = meta['image_name']
        match = re.search(r'(\d+)', str(name))
        if match:
            return int(match.group(1))

    # 尝试从 dataparser 的 image_filename 解析
    if image_filename:
        stem = Path(str(image_filename)).stem
        match = re.search(r'(\d+)', stem)
        if match:
            return int(match.group(1))

    # 尝试从 render 的 image_name 解析
    if image_name:
        stem = Path(str(image_name)).name
        match = re.search(r'(\d+)', stem)
        if match:
            return int(match.group(1))
    
    # 没有找到，使用fallback
    return fallback

def load_mask(mask_path: Path, size: tuple) -> torch.Tensor:
    """加载mask并resize到目标尺寸"""
    img = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    arr = np.array(img)
    return torch.from_numpy(arr >= 1)

def select_gaussians_from_mask(mask: torch.Tensor, info: dict, num_total: int) -> set:
    """
    从rasterization info中选出mask覆盖的高斯
    核心逻辑：tile-based voting
    """
    if info is None:
        raise ValueError("需要rasterization info")
    
    # 提取info
    isect_offsets = info["isect_offsets"].reshape(-1)
    flatten_ids = info["flatten_ids"]
    tile_size = int(info["tile_size"].item() if isinstance(info["tile_size"], torch.Tensor) else info["tile_size"])
    tile_width = int(info["tile_width"].item() if isinstance(info["tile_width"], torch.Tensor) else info["tile_width"])
    
    # 计算哪些tile被mask覆盖
    H, W = mask.shape
    tile_h = (H + tile_size - 1) // tile_size
    tile_w = tile_width
    
    # Pad mask到tile边界
    pad_h = max(0, tile_h * tile_size - H)
    pad_w = max(0, tile_w * tile_size - W)
    m = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else mask
    
    # 重组为tile grid并检测覆盖
    m = m.view(tile_h, tile_size, tile_w, tile_size)
    tile_hit = m.any(dim=3).any(dim=1)  # [tile_h, tile_w]
    hit_tiles = tile_hit.flatten().nonzero(as_tuple=False).squeeze(1)
    
    if hit_tiles.numel() == 0:
        return set()
    
    # 收集这些tile中的所有高斯ID
    selected = set()
    for tile_idx in hit_tiles.tolist():
        start = int(isect_offsets[tile_idx].item())
        end = int(isect_offsets[tile_idx + 1].item()) if tile_idx + 1 < isect_offsets.numel() else flatten_ids.numel()
        if end <= start:
            continue
        ids = (flatten_ids[start:end] % num_total).long()
        selected.update(ids.tolist())
    
    return selected

def render_and_save(model, camera, output_path: Path):
    """渲染并保存图像"""
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
    
    # 提取RGB
    if "rgb" in outputs:
        img = outputs["rgb"]
    elif "image" in outputs:
        img = outputs["image"]
    elif "comp_rgb" in outputs:
        img = outputs["comp_rgb"]
    else:
        raise KeyError(f"无RGB输出: {list(outputs.keys())}")
    
    # 转numpy并保存
    if img.dim() == 4:
        img = img[0]
    if img.shape[0] == 3:  # CHW → HWC
        img = img.permute(1, 2, 0)
    
    img_np = (img.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img_np).save(output_path)
    
    # 返回info用于选高斯
    return outputs.get("info")  # 如果outputs包含info的话

def patch_rasterization_to_capture_info():
    """Monkey patch gsplat.rasterization来捕获info"""
    captured = {}
    
    try:
        import gsplat_original.rendering as gsplat_rendering
        original_fn = gsplat_rendering.rasterization
        
        def patched(*args, **kwargs):
            render, alpha, info = original_fn(*args, **kwargs)
            captured['info'] = info
            return render, alpha, info
        
        gsplat_rendering.rasterization = patched
        print("✅ Patched gsplat_original.rasterization")
        return captured
    except ImportError:
        raise ImportError("找不到gsplat_original")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--mask-template", default="car_mask_001/id1_{frame_idx:02d}_mask.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-cam-idx", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    info_holder = patch_rasterization_to_capture_info()
    
    print("=" * 70)
    print("加载模型...")
    config, pipeline, _, step = eval_setup(
        args.config,
        test_mode="inference",
        update_config_callback=lambda cfg: _override_config(cfg, args.checkpoint),
    )
    pipeline.to(device)
    model = pipeline.model
    model.eval()
    
    num_gauss = model.means.shape[0]
    print(f"总高斯数: {num_gauss}")
    
    cameras = pipeline.datamanager.train_dataset.cameras.to(device)
    print(f"数据集大小: {len(cameras)} cameras")
    dataparser_outputs = getattr(pipeline.datamanager, "train_dataparser_outputs", None)
    images_root = None
    if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
        images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
    
    # ✅ 诊断：检查前几帧的metadata
    print(f"\n{'='*70}")
    print("[诊断] 检查前3帧的metadata和帧号提取:")
    diag_count = 0
    for idx in range(len(cameras)):
        cam = cameras[idx:idx+1]
        meta = cam.metadata or {}
        
        cam_idx = None
        for k in ["sensor_idxs", "sensor_idx", "cam_idx"]:
            if k in meta:
                v = meta[k]
                cam_idx = int(v.item() if isinstance(v, torch.Tensor) else v)
                break
        
        if cam_idx != args.target_cam_idx:
            continue
        
        image_filename = None
        image_name = None
        if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
            image_filename = dataparser_outputs.image_filenames[idx]
            if images_root is not None:
                try:
                    image_name = str(Path(image_filename).with_suffix("").relative_to(images_root))
                except ValueError:
                    image_name = str(Path(image_filename).with_suffix("").name)
            else:
                image_name = str(Path(image_filename).with_suffix("").name)

        real_idx = extract_real_frame_idx(meta, diag_count, image_filename=image_filename, image_name=image_name)
        print(f"\n处理帧 {diag_count} (camera_idx={idx}):")
        print(f"  提取的真实帧号: {real_idx}")
        print(f"  metadata keys: {list(meta.keys())}")
        if image_name:
            print(f"  image_name: {image_name}")
        
        diag_count += 1
        if diag_count >= 3:
            break
    
    print(f"{'='*70}\n")
    
    # 备份原始features
    original_dc = model.features_dc.data.clone()
    original_rest = model.features_rest.data.clone()
    
    # 处理每一帧
    processed = 0
    for idx in range(len(cameras)):
        cam = cameras[idx:idx+1]
        meta = cam.metadata or {}
        
        cam_idx = None
        for k in ["sensor_idxs", "sensor_idx", "cam_idx"]:
            if k in meta:
                v = meta[k]
                cam_idx = int(v.item() if isinstance(v, torch.Tensor) else v)
                break
        
        if cam_idx != args.target_cam_idx:
            continue
        
        # ✅ 提取真实帧号
        image_filename = None
        image_name = None
        if dataparser_outputs is not None and getattr(dataparser_outputs, "image_filenames", None):
            image_filename = dataparser_outputs.image_filenames[idx]
            if images_root is not None:
                try:
                    image_name = str(Path(image_filename).with_suffix("").relative_to(images_root))
                except ValueError:
                    image_name = str(Path(image_filename).with_suffix("").name)
            else:
                image_name = str(Path(image_filename).with_suffix("").name)

        real_frame_idx = extract_real_frame_idx(meta, processed, image_filename=image_filename, image_name=image_name)
        
        W, H = int(cam.width.item()), int(cam.height.item())
        print(f"\n{'='*60}")
        print(f"处理帧 {processed} → 原始帧 {real_frame_idx:02d} (camera_idx={idx})")
        if image_name:
            print(f"  image_name: {image_name}")
        
        # 渲染原始图像
        print("  渲染原始图像...")
        info_holder.clear()
        render_and_save(model, cam, 
                       args.output_dir / f"frame{processed:04d}_real{real_frame_idx:02d}_original.png")
        info_raster = info_holder.get('info')
        
        if info_raster is None:
            print("  ❌ 无法获取info")
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
            continue
        
        # ✅ 使用真实帧号加载mask
        mask_path = args.mask_root / args.mask_template.format(frame_idx=real_frame_idx)
        
        if not mask_path.exists():
            print(f"  ⚠️  Mask不存在: {mask_path.name}")
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
            continue
        
        mask = load_mask(mask_path, (W, H)).to(device)
        print(f"  ✅ 加载mask: {mask_path.name}, {mask.sum().item():.0f} pixels")
        
        if mask.sum() == 0:
            print("  ⚠️  空mask")
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
            continue
        
        # 选择高斯
        selected_ids = select_gaussians_from_mask(mask, info_raster, num_gauss)
        print(f"  选中 {len(selected_ids)} 个高斯")
        
        if len(selected_ids) == 0:
            print("  ⚠️  未选中任何高斯")
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
            continue
        
        # 临时变黑
        # ids_tensor = torch.tensor(list(selected_ids), dtype=torch.long, device=device)
        # with torch.no_grad():
        #     model.opacities.data[ids_tensor] = -10.0  # 或更小
        #     model.features_dc.data[ids_tensor] = -225.0
        #     model.features_rest.data[ids_tensor] = -225.0
        
        # 渲染masked
        render_and_save(model, cam,
                       args.output_dir / f"frame{processed:04d}_real{real_frame_idx:02d}_masked.png")
        print("  ✅ 渲染完成")
        
        # 复原
        with torch.no_grad():
            model.features_dc.data.copy_(original_dc)
            model.features_rest.data.copy_(original_rest)
        
        processed += 1
        if args.max_frames and processed >= args.max_frames:
            break
    
    print(f"\n{'='*70}")
    print(f"✅ 完成！处理了 {processed} 帧")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()
