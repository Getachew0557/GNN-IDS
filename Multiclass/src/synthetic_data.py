import numpy as np
import torch
import os
import random

def set_seed(seed: int = 40) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim, time_steps=3):
    set_seed()
    action_node_idx = list(action_nodes.keys())
    num_action_nodes = len(action_node_idx)
    X = torch.zeros(num_samples, num_nodes, rt_meas_dim * time_steps)
    Y = torch.zeros(num_samples, num_nodes, dtype=torch.long)  # Long for multiclass labels
    sd = 0.2
    for t in range(time_steps):
        rt_measurements = []
        for i in range(rt_meas_dim // 3):
            mu = np.random.uniform(0.3, 0.3)
            lambda_p = np.random.uniform(3.0, 3.0)
            rt_1 = torch.normal(mu, sd, size=(num_samples, num_action_nodes))
            rt_2 = torch.poisson(torch.ones(num_samples, num_action_nodes) * lambda_p)
            rt_3 = rt_1.abs() ** 0.5 + rt_2 * 0.5 + 0.5
            rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
        rt_measurements = torch.cat(rt_measurements, dim=2)
        X[:, action_node_idx, t * rt_meas_dim:(t + 1) * rt_meas_dim] = rt_measurements
    return X, Y

def malic_data(num_samples, num_nodes, action_nodes, rt_meas_dim, time_steps=3):
    action_node_idx = list(action_nodes.keys())
    attack_types = ['Web Attack - Brute Force', 'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'DDoS', 'Bot', 'PortScan']
    num_action_nodes = len(action_node_idx)
    X = torch.zeros(num_samples * num_action_nodes, num_nodes, rt_meas_dim * time_steps)
    Y = torch.zeros(num_samples * num_action_nodes, num_nodes, dtype=torch.long)  # Long for multiclass labels

    for idx, node_idx in enumerate(action_node_idx):
        attack_label = idx + 1  # Labels 1 to 7 for attack types
        x, y = benign_data(num_samples, num_nodes, action_nodes, rt_meas_dim, time_steps)
        action_name = action_nodes[node_idx]['predicate']
        for t in range(time_steps):
            mali_meas = sample_mali_scen(num_samples, 1, rt_meas_dim, action_name, t, attack_label)
            x[:, [node_idx], t * rt_meas_dim:(t + 1) * rt_meas_dim] = mali_meas
        y[:, node_idx] = attack_label
        X[idx * num_samples:(idx + 1) * num_samples, :, :] = x
        Y[idx * num_samples:(idx + 1) * num_samples, :] = y
    return X, Y

def sample_mali_scen(num_samples, num_comp_nodes, rt_meas_dim, action_name, time_step, attack_label):
    set_seed()
    rt_measurements = []
    # Define attack-specific parameters
    attack_params = {
        1: {'mu_range': (0.3, 0.3), 'lambda_range': (1, 5), 'sd': 0.2},  # Web Attack - Brute Force
        2: {'mu_range': (0.1, 0.5), 'lambda_range': (3.0, 3.0), 'sd': 0.1},  # DoS slowloris
        3: {'mu_range': (0.2, 0.4), 'lambda_range': (2, 4), 'sd': 0.15},  # FTP-Patator
        4: {'mu_range': (0.15, 0.35), 'lambda_range': (1.5, 3.5), 'sd': 0.12},  # SSH-Patator
        5: {'mu_range': (0.1, 0.6), 'lambda_range': (2, 6), 'sd': 0.18},  # DDoS
        6: {'mu_range': (0.25, 0.45), 'lambda_range': (1, 3), 'sd': 0.13},  # Bot
        7: {'mu_range': (0.05, 0.55), 'lambda_range': (2.5, 4.5), 'sd': 0.17}  # PortScan
    }
    params = attack_params.get(attack_label, {'mu_range': (0.1, 0.5), 'lambda_range': (3.0, 3.0), 'sd': 0.1})
    for i in range(rt_meas_dim // 3):
        mu = np.random.uniform(*params['mu_range'])
        lambda_p = np.random.uniform(*params['lambda_range'])
        rt_1 = torch.normal(mu, params['sd'], size=(num_samples, num_comp_nodes))
        rt_2 = torch.poisson(torch.ones(num_samples, num_comp_nodes) * lambda_p)
        rt_3 = rt_1.abs() ** 0.5 + rt_2 * 0.5 + 0.5 + (time_step * 0.1)
        rt_measurements.append(torch.stack((rt_1, rt_2, rt_3), dim=2))
    return torch.cat(rt_measurements, dim=2)

def oversample_malicious(X_malic, Y_malic, target_samples, num_nodes, rt_meas_dim, time_steps=3):
    current_samples = X_malic.shape[0]
    if current_samples >= target_samples:
        return X_malic, Y_malic
    additional_samples = target_samples - current_samples
    X_aug = torch.zeros(additional_samples, num_nodes, rt_meas_dim * time_steps)
    Y_aug = torch.zeros(additional_samples, num_nodes, dtype=torch.long)
    for i in range(additional_samples):
        idx1, idx2 = np.random.choice(current_samples, 2)
        alpha = np.random.beta(0.5, 0.5)
        X_aug[i] = alpha * X_malic[idx1] + (1 - alpha) * X_malic[idx2]
        Y_aug[i] = Y_malic[idx1]  # Use one of the labels (mixup for features only)
    return torch.cat([X_malic, X_aug], dim=0), torch.cat([Y_malic, Y_aug], dim=0)

def gene_dataset(num_benign, num_malic, num_nodes, action_nodes, rt_meas_dim, time_steps=3):
    X_benign, Y_benign = benign_data(num_benign, num_nodes, action_nodes, rt_meas_dim, time_steps)
    X_malic, Y_malic = malic_data(num_malic, num_nodes, action_nodes, rt_meas_dim, time_steps)
    action_mask = list(action_nodes.keys())
    X_malic, Y_malic = oversample_malicious(X_malic, Y_malic, num_malic * 2, num_nodes, rt_meas_dim, time_steps)
    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)[:, action_mask]
    return X, Y