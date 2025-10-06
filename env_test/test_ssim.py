# test_ssim.py
import torch
from fused_ssim import fused_ssim

print("=== Testing fused-ssim ===")

# 创建两个假图像
img1 = torch.rand(1, 3, 256, 256, device="cuda")
img2 = img1 + torch.randn_like(img1) * 0.1  # 加点噪声

try:
    ssim_value = fused_ssim(img1, img2)
    print(f"✓ fused-ssim works")
    print(f"  SSIM value: {ssim_value.item():.4f}")
    
    # 测试梯度
    img1.requires_grad = True
    ssim_value = fused_ssim(img1, img2)
    ssim_value.backward()
    print(f"✓ Gradient computation works")
    print(f"  Grad shape: {img1.grad.shape}")
    
except Exception as e:
    print(f"✗ fused-ssim failed: {e}")

print("\n=== fused-ssim test completed ===")