# test_gsplat.py
import torch
import gsplat

print("=== Testing gsplat core functions ===")

# 创建一些假的高斯参数
num_points = 100
means = torch.randn(num_points, 3, device="cuda")
scales = torch.rand(num_points, 3, device="cuda") * 0.1
quats = torch.randn(num_points, 4, device="cuda")
quats = quats / quats.norm(dim=-1, keepdim=True)  # 归一化
colors = torch.rand(num_points, 3, device="cuda")
opacities = torch.rand(num_points, 1, device="cuda")

# 创建假的相机参数
viewmat = torch.eye(4, device="cuda")
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], device="cuda", dtype=torch.float32)
width, height = 640, 480

print(f"Created {num_points} Gaussians")
print(f"Image size: {width}x{height}")

try:
    # 测试光栅化 (这是 SplatAD 的核心)
    from gsplat import rasterization
    
    rendered = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
    )
    
    print(f"✓ Rasterization successful")
    print(f"  Output shape: {rendered[0].shape}")
    print(f"  Output range: [{rendered[0].min():.3f}, {rendered[0].max():.3f}]")
    
except Exception as e:
    print(f"✗ Rasterization failed: {e}")

print("\n=== gsplat test completed ===")