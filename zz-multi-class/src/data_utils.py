"""
Data Utilities for Attack Graph Classification

This module handles data loading, preprocessing, and dataset generation
for both synthetic and public cybersecurity datasets.
"""

"""
Data Utilities for Attack Graph Classification

This module handles data loading, preprocessing, and dataset generation
for both synthetic and public cybersecurity datasets.
"""

import pandas as pd
import numpy as np
import torch
import os
import random
from typing import Tuple, Optional, Dict, Any, List  # Added List import
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_CICIDS(num_benign: int, action_node_idx: List[int]) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Load and preprocess CIC-IDS-2017 dataset for multiclass classification.
    
    Args:
        num_benign: Number of benign samples to use
        action_node_idx: List of action node indices
        
    Returns:
        Tuple of (x_benign, y_benign, x_malic, y_malic) tensors
    """
    csv_file = '../data/public/CICD-IDS2017.csv'
    if not os.path.exists(csv_file):
        print(f'Dataset file not found: {csv_file}')
        return None, None, None, None

    try:
        # Load dataset with memory optimization
        df = pd.read_csv(csv_file, low_memory=False)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        x = scaler.fit_transform(x)
        x = torch.from_numpy(x).float()

        # Define attack types and their distributions
        attack_types = [
            'Web Attack - Brute Force', 'DoS slowloris', 'FTP-Patator', 
            'SSH-Patator', 'DDoS', 'Bot', 'PortScan'
        ]
        num_action_nodes = len(action_node_idx)
        
        # Varied distribution mimicking real-world proportions
        num_malic_per_class = [8500, 9200, 10500, 11800, 8000, 9500, 11200]
        
        # Process benign samples
        total_benign = num_benign * num_action_nodes
        x_benign = x[y == 'BENIGN'][:total_benign].reshape(-1, num_action_nodes, x.shape[1])
        y_benign = torch.zeros(x_benign.shape[0], x_benign.shape[1], dtype=torch.long)

        # Process malicious samples
        total_malic_samples = sum(num_malic_per_class)
        x_malic = torch.zeros(total_malic_samples, num_action_nodes, x.shape[1])
        y_malic = torch.zeros(total_malic_samples, num_action_nodes, dtype=torch.long)

        start_idx = 0
        for idx, val in enumerate(action_node_idx):
            attack = attack_types[idx]
            attack_label = idx + 1  # Labels 1 to 7
            num_malic = num_malic_per_class[idx]
            
            attack_data = x[y == attack]
            num_available = attack_data.shape[0]
            print(f"{attack}: {num_available} available samples, requesting {num_malic}")
            
            # Handle class imbalance with oversampling if needed
            if num_available < num_malic:
                print(f"Warning: Only {num_available} samples for {attack}. Oversampling to {num_malic}.")
                indices = np.random.choice(num_available, size=num_malic, replace=True)
                attack_samples = attack_data[indices]
            else:
                attack_samples = attack_data[:num_malic]
            
            # Assign samples to corresponding action node
            end_idx = start_idx + num_malic
            x_malic[start_idx:end_idx, idx, :] = attack_samples
            y_malic[start_idx:end_idx, idx] = attack_label
            start_idx = end_idx

        return x_benign, y_benign, x_malic, y_malic
        
    except Exception as e:
        print(f"Error loading CICIDS dataset: {e}")
        return None, None, None, None


def gene_public_dataset(action_node_idx: List[int], num_nodes: int, num_benign: int, 
                       num_malic: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Dataset 2 using public CIC-IDS-2017 data with multiclass distribution.
    
    Args:
        action_node_idx: List of action node indices
        num_nodes: Total number of nodes in graph
        num_benign: Number of benign samples
        num_malic: Number of malicious samples (ignored, uses predefined distribution)
        
    Returns:
        Tuple of (X, Y) tensors
    """
    num_action_nodes = len(action_node_idx)
    x_benign, y_benign, x_malic, y_malic = load_CICIDS(num_benign, action_node_idx)
    
    if x_benign is None:
        return None, None
        
    rt_meas_dim = x_benign.shape[2]
    
    # Create full graph tensors
    X_benign = torch.zeros(num_benign, num_nodes, rt_meas_dim)
    X_benign[:, action_node_idx, :] = x_benign
    Y_benign = y_benign

    X_malic = torch.zeros(x_malic.shape[0], num_nodes, rt_meas_dim)
    X_malic[:, action_node_idx, :] = x_malic
    Y_malic = y_malic

    # Combine benign and malicious samples
    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)

    return X, Y


def generate_synthetic_benign_data(num_samples: int, num_nodes: int, action_nodes: Dict[int, Any], 
                                  rt_meas_dim: int, time_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic benign data simulating normal network behavior.
    
    Args:
        num_samples: Number of samples to generate
        num_nodes: Total number of nodes
        action_nodes: Dictionary of action nodes
        rt_meas_dim: Runtime measurement dimension
        time_steps: Number of temporal steps
        
    Returns:
        Tuple of (X, Y) tensors
    """
    set_seed()
    action_node_idx = list(action_nodes.keys())
    num_action_nodes = len(action_node_idx)
    
    X = torch.zeros(num_samples, num_nodes, rt_meas_dim * time_steps)
    Y = torch.zeros(num_samples, num_nodes, dtype=torch.long)
    
    # Generate realistic benign patterns
    sd = 0.2
    for t in range(time_steps):
        rt_measurements = []
        for i in range(rt_meas_dim // 3):
            mu = np.random.uniform(0.3, 0.3)
            lambda_p = np.random.uniform(3.0, 3.0)
            
            # Simulate different types of network measurements
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_action_nodes))  # Continuous metrics
            rt_2 = torch.poisson(torch.ones(num_samples, num_action_nodes) * lambda_p)  # Count metrics
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * 0.5 + 0.5  # Derived metrics
            
            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
        
        rt_measurements = torch.cat(rt_measurements, dim=2)
        X[:, action_node_idx, t * rt_meas_dim:(t + 1) * rt_meas_dim] = rt_measurements
    
    return X, Y


def generate_synthetic_malicious_data(num_samples: int, num_nodes: int, action_nodes: Dict[int, Any], 
                                     rt_meas_dim: int, time_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic malicious data simulating various attack patterns.
    
    Args:
        num_samples: Number of samples per attack type
        num_nodes: Total number of nodes
        action_nodes: Dictionary of action nodes
        rt_meas_dim: Runtime measurement dimension
        time_steps: Number of temporal steps
        
    Returns:
        Tuple of (X, Y) tensors
    """
    action_node_idx = list(action_nodes.keys())
    attack_types = [
        'Web Attack - Brute Force', 'DoS slowloris', 'FTP-Patator', 
        'SSH-Patator', 'DDoS', 'Bot', 'PortScan'
    ]
    num_action_nodes = len(action_node_idx)
    
    X = torch.zeros(num_samples * num_action_nodes, num_nodes, rt_meas_dim * time_steps)
    Y = torch.zeros(num_samples * num_action_nodes, num_nodes, dtype=torch.long)

    # Generate data for each attack type
    for idx, node_idx in enumerate(action_node_idx):
        attack_label = idx + 1  # Labels 1 to 7 for attack types
        
        # Start with benign pattern and inject attack signatures
        x, y = generate_synthetic_benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim, time_steps)
        action_name = action_nodes[node_idx]['predicate']
        
        # Modify measurements to reflect attack behavior
        for t in range(time_steps):
            mali_meas = sample_malicious_scenario(num_samples, 1, rt_meas_dim, action_name, t, attack_label)
            x[:, [node_idx], t * rt_meas_dim:(t + 1) * rt_meas_dim] = mali_meas
        
        y[:, node_idx] = attack_label
        X[idx * num_samples:(idx + 1) * num_samples, :, :] = x
        Y[idx * num_samples:(idx + 1) * num_samples, :] = y
    
    return X, Y


def sample_malicious_scenario(num_samples: int, num_comp_nodes: int, rt_meas_dim: int, 
                             action_name: str, time_step: int, attack_label: int) -> torch.Tensor:
    """
    Generate malicious measurements for specific attack scenarios.
    
    Args:
        num_samples: Number of samples
        num_comp_nodes: Number of compromised nodes
        rt_meas_dim: Runtime measurement dimension
        action_name: Name of the attack action
        time_step: Temporal step
        attack_label: Attack type label
        
    Returns:
        Tensor of malicious measurements
    """
    set_seed()
    rt_measurements = []
    
    # Attack-specific parameter configurations
    attack_params = {
        1: {'mu_range': (0.3, 0.3), 'lambda_range': (1, 5), 'sd': 0.2},    # Web Attack - Brute Force
        2: {'mu_range': (0.1, 0.5), 'lambda_range': (3.0, 3.0), 'sd': 0.1}, # DoS slowloris
        3: {'mu_range': (0.2, 0.4), 'lambda_range': (2, 4), 'sd': 0.15},    # FTP-Patator
        4: {'mu_range': (0.15, 0.35), 'lambda_range': (1.5, 3.5), 'sd': 0.12}, # SSH-Patator
        5: {'mu_range': (0.1, 0.6), 'lambda_range': (2, 6), 'sd': 0.18},    # DDoS
        6: {'mu_range': (0.25, 0.45), 'lambda_range': (1, 3), 'sd': 0.13},  # Bot
        7: {'mu_range': (0.05, 0.55), 'lambda_range': (2.5, 4.5), 'sd': 0.17} # PortScan
    }
    
    params = attack_params.get(attack_label, {'mu_range': (0.1, 0.5), 'lambda_range': (3.0, 3.0), 'sd': 0.1})
    
    for i in range(rt_meas_dim // 3):
        mu = np.random.uniform(*params['mu_range'])
        lambda_p = np.random.uniform(*params['lambda_range'])
        
        # Generate attack-specific patterns
        rt_1 = torch.normal(mu, params['sd'], size=(num_samples, num_comp_nodes))
        rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes) * lambda_p)
        rt_3 = rt_1.abs() ** 0.5 + rt_2 * 0.5 + 0.5 + (time_step * 0.1)  # Temporal evolution
        
        rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    
    return torch.cat(rt_measurements, dim=2)


def oversample_malicious(X_malic: torch.Tensor, Y_malic: torch.Tensor, target_samples: int, 
                        num_nodes: int, rt_meas_dim: int, time_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Oversample malicious samples using mixup augmentation to handle class imbalance.
    
    Args:
        X_malic: Malicious feature tensor
        Y_malic: Malicious label tensor
        target_samples: Target number of samples
        num_nodes: Number of nodes
        rt_meas_dim: Runtime measurement dimension
        time_steps: Number of temporal steps
        
    Returns:
        Tuple of augmented (X, Y) tensors
    """
    current_samples = X_malic.shape[0]
    if current_samples >= target_samples:
        return X_malic, Y_malic
        
    additional_samples = target_samples - current_samples
    X_aug = torch.zeros(additional_samples, num_nodes, rt_meas_dim * time_steps)
    Y_aug = torch.zeros(additional_samples, num_nodes, dtype=torch.long)
    
    # Mixup augmentation for generating synthetic samples
    for i in range(additional_samples):
        idx1, idx2 = np.random.choice(current_samples, 2)
        alpha = np.random.beta(0.5, 0.5)  # Mixup parameter
        X_aug[i] = alpha * X_malic[idx1] + (1 - alpha) * X_malic[idx2]
        Y_aug[i] = Y_malic[idx1]  # Use one of the labels (mixup for features only)
    
    return torch.cat([X_malic, X_aug], dim=0), torch.cat([Y_malic, Y_aug], dim=0)


def gene_synthetic_dataset(num_benign: int, num_malic: int, num_nodes: int, 
                          action_nodes: Dict[int, Any], rt_meas_dim: int, 
                          time_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate complete synthetic dataset with benign and malicious samples.
    
    Args:
        num_benign: Number of benign samples
        num_malic: Number of malicious samples per class
        num_nodes: Total number of nodes
        action_nodes: Dictionary of action nodes
        rt_meas_dim: Runtime measurement dimension
        time_steps: Number of temporal steps
        
    Returns:
        Tuple of (X, Y) tensors
    """
    # Generate base datasets
    X_benign, Y_benign = generate_synthetic_benign_data(num_benign, num_nodes, action_nodes, rt_meas_dim, time_steps)
    X_malic, Y_malic = generate_synthetic_malicious_data(num_malic, num_nodes, action_nodes, rt_meas_dim, time_steps)
    
    action_mask = list(action_nodes.keys())
    
    # Apply oversampling if needed
    X_malic, Y_malic = oversample_malicious(X_malic, Y_malic, num_malic * 2, num_nodes, rt_meas_dim, time_steps)
    
    # Combine datasets
    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)[:, action_mask]
    
    return X, Y


def prepare_dataset(X: torch.Tensor, Y: torch.Tensor, test_size: float = 0.2, 
                   val_size: float = 0.15, random_state: int = 42) -> Tuple:
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Args:
        X: Feature tensor
        Y: Label tensor
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Use first action node for stratification
    Y_stratify = Y[:, 0] if Y.dim() > 1 else Y
    
    # Split into train+val and test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y_stratify
    )
    
    # Split train+val into train and val
    Y_stratify_train = Y_train_val[:, 0] if Y_train_val.dim() > 1 else Y_train_val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=Y_stratify_train
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test