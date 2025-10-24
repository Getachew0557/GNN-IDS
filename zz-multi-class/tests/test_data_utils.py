"""
Tests for data utilities.
"""

import pytest
import torch
import numpy as np
from src.data_utils import set_seed, generate_synthetic_benign_data, prepare_dataset


class TestDataUtils:
    """Test cases for data utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Test numpy randomness
        np_val1 = np.random.randn()
        set_seed(42)
        np_val2 = np.random.randn()
        assert np_val1 == np_val2
        
        # Test torch randomness
        torch_val1 = torch.randn(1)
        set_seed(42)
        torch_val2 = torch.randn(1)
        assert torch.allclose(torch_val1, torch_val2)
    
    def test_generate_synthetic_benign_data(self):
        """Test synthetic benign data generation."""
        num_samples = 10
        num_nodes = 5
        action_nodes = {0: {'predicate': 'test', 'attributes': [], 'shape': 'diamond'}}
        rt_meas_dim = 6  # Small dimension for testing
        
        X, Y = generate_synthetic_benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim)
        
        assert X.shape == (num_samples, num_nodes, rt_meas_dim)
        assert Y.shape == (num_samples, num_nodes)
        assert torch.is_tensor(X)
        assert torch.is_tensor(Y)
        # Action node should have non-zero features
        assert not torch.allclose(X[:, 0, :], torch.zeros(rt_meas_dim))
        # Non-action nodes should have zero features
        assert torch.allclose(X[:, 1, :], torch.zeros(rt_meas_dim))
    
    def test_prepare_dataset(self):
        """Test dataset splitting."""
        # Create sample data
        num_samples = 100
        num_nodes = 5
        feature_dim = 10
        
        X = torch.randn(num_samples, num_nodes, feature_dim)
        Y = torch.randint(0, 2, (num_samples, num_nodes))
        
        # Split dataset
        X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_dataset(X, Y)
        
        # Check shapes
        total_samples = num_samples
        test_size = int(total_samples * 0.2)
        val_size = int((total_samples - test_size) * 0.15)
        train_size = total_samples - test_size - val_size
        
        assert X_train.shape[0] == train_size
        assert X_val.shape[0] == val_size
        assert X_test.shape[0] == test_size
        
        # Check no data leakage
        combined = torch.cat([X_train, X_val, X_test], dim=0)
        assert combined.shape[0] == num_samples
        
        # Check stratification (roughly similar class distribution)
        train_class_dist = torch.unique(Y_train, return_counts=True)[1].float()
        val_class_dist = torch.unique(Y_val, return_counts=True)[1].float()
        test_class_dist = torch.unique(Y_test, return_counts=True)[1].float()
        
        # Distributions should be roughly similar
        assert torch.allclose(train_class_dist / train_class_dist.sum(), 
                             val_class_dist / val_class_dist.sum(), 
                             atol=0.2)


if __name__ == "__main__":
    pytest.main([__file__])