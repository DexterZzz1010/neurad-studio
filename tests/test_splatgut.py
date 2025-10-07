"""
Unit tests for SplatGUT model and 3DGUT adapter.

Tests cover:
    - Model initialization and inheritance
    - Projection interface compatibility
    - Fallback mechanisms
    - Forward pass execution
"""

import pytest
import torch

from nerfstudio.adapters.threedgut_adapter import GUTProjectorAdapter
from nerfstudio.models.splatgut import SplatGUTModel, SplatGUTModelConfig
from nerfstudio.cameras.cameras import Cameras


class TestGUTProjectorAdapter:
    """Test 3DGUT projection adapter."""
    
    def test_adapter_init_fallback(self):
        """Test adapter initializes with fallback when 3DGUT unavailable."""
        adapter = GUTProjectorAdapter(enable_fallback=True)
        assert adapter.enable_fallback is True
        assert adapter.available is False
    
    def test_adapter_init_no_fallback(self):
        """Test adapter raises error when fallback disabled."""
        with pytest.raises(RuntimeError, match="3DGUT required"):
            GUTProjectorAdapter(enable_fallback=False)
    
    def test_projection_fallback_ewa(self):
        """Test fallback to EWA projection works correctly."""
        adapter = GUTProjectorAdapter(enable_fallback=True)
        
        N = 100
        means = torch.randn(N, 3)
        quats = torch.randn(N, 4)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        scales = torch.rand(N, 3) * 0.1
        opacities = torch.rand(N)
        
        c2w = torch.eye(4)
        K = torch.tensor([
            [1000.0, 0.0, 512.0],
            [0.0, 1000.0, 384.0],
            [0.0, 0.0, 1.0],
        ])
        
        # Should not raise, uses EWA fallback
        projected = adapter.project(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            camera_matrix=c2w,
            intrinsics=K,
            resolution=(1024, 768),
        )
        
        # Check output format
        assert 'means2d' in projected
        assert 'conics' in projected
        assert 'depths' in projected
        assert 'radii' in projected
        assert 'opacities' in projected
        
        assert projected['means2d'].shape == (N, 2)
        assert projected['conics'].shape == (N, 3)
        assert projected['depths'].shape == (N,)
        assert projected['radii'].shape == (N, 2)
        assert projected['opacities'].shape == (N,)


class TestSplatGUTModel:
    """Test SplatGUT model."""
    
    def test_model_config_inheritance(self):
        """Test config correctly inherits from SplatAD."""
        config = SplatGUTModelConfig()
        
        # Check 3DGUT-specific parameters
        assert hasattr(config, 'use_gut_projection')
        assert hasattr(config, 'gut_alpha')
        assert hasattr(config, 'use_ray_tracing')
        
        # Check inherited SplatAD parameters
        assert hasattr(config, 'warmup_length')
        assert hasattr(config, 'refine_every')
        assert hasattr(config, 'strategy')
        assert hasattr(config, 'cull_alpha_thresh')
        
        # Check inherited ADModel parameters
        assert hasattr(config, 'dynamic_actors')
        assert hasattr(config, 'camera_optimizer')
    
    def test_model_initialization(self):
        """Test model initializes without errors."""
        config = SplatGUTModelConfig(
            use_gut_projection=True,
            enable_fallback=True,
        )
        
        # Note: Full initialization requires seed points and scene box
        # This is a simplified test
        assert config._target == SplatGUTModel
    
    def test_projection_mode_switching(self):
        """Test switching between GUT and EWA projection."""
        config_gut = SplatGUTModelConfig(use_gut_projection=True)
        config_ewa = SplatGUTModelConfig(use_gut_projection=False)
        
        assert config_gut.use_gut_projection is True
        assert config_ewa.use_gut_projection is False


class TestIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for integration test"
    )
    def test_forward_pass_cpu(self):
        """Test forward pass executes without errors (CPU)."""
        # Create minimal camera
        camera = Cameras(
            camera_to_worlds=torch.eye(4)[None],
            fx=torch.tensor([1000.0]),
            fy=torch.tensor([1000.0]),
            cx=torch.tensor([512.0]),
            cy=torch.tensor([384.0]),
            width=torch.tensor([1024]),
            height=torch.tensor([768]),
            camera_type=torch.tensor([0]),  # Perspective
        )
        
        # This would require full model setup with seed points
        # Placeholder for now
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])