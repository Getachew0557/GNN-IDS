import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler

def load_CICIDS(num_benign, action_node_idx):
    csv_file = '../datasets/public/CICIDS-2017.csv'
    if not os.path.exists(csv_file):
        print('The dataset file does not exist.')
        return None, None, None, None

    df = pd.read_csv(csv_file, low_memory=False)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    x = torch.from_numpy(x).float()

    attack_types = ['Web Attack - Brute Force', 'DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'DDoS', 'Bot', 'PortScan']
    num_action_nodes = len(action_node_idx)
    
    # Varied distribution for each malicious class (8000-12000 labels per class, based on CICIDS proportions)
    num_malic_per_class = [8500, 9200, 10500, 11800, 8000, 9500, 11200]  # Scaled from actual CICIDS counts
    
    total_benign = num_benign * num_action_nodes
    x_benign = x[y == 'BENIGN'][:total_benign].reshape(-1, num_action_nodes, x.shape[1])
    y_benign = torch.zeros(x_benign.shape[0], x_benign.shape[1], dtype=torch.long)

    # Calculate total malicious samples
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
        
        if num_available < num_malic:
            # Oversample by random duplication to reach num_malic
            print(f"Warning: Only {num_available} samples for {attack}. Oversampling to {num_malic}.")
            indices = np.random.choice(num_available, size=num_malic, replace=True)
            attack_samples = attack_data[indices]
        else:
            attack_samples = attack_data[:num_malic]
        
        # Fill the slice for this attack type
        end_idx = start_idx + num_malic
        x_malic[start_idx:end_idx, idx, :] = attack_samples
        y_malic[start_idx:end_idx, idx] = attack_label
        start_idx = end_idx

    return x_benign, y_benign, x_malic, y_malic

def gene_dataset(action_node_idx, num_nodes, num_benign, num_malic=None):
    """
    Generate Dataset 2 with varied multiclass distribution.
    num_malic is ignored; uses predefined num_malic_per_class in load_CICIDS.
    """
    num_action_nodes = len(action_node_idx)
    x_benign, y_benign, x_malic, y_malic = load_CICIDS(num_benign, action_node_idx)
    if x_benign is None:
        return None, None
    rt_meas_dim = x_benign.shape[2]
    X_benign = torch.zeros(num_benign, num_nodes, rt_meas_dim)
    X_benign[:, action_node_idx, :] = x_benign
    Y_benign = y_benign

    X_malic = torch.zeros(x_malic.shape[0], num_nodes, rt_meas_dim)
    X_malic[:, action_node_idx, :] = x_malic
    Y_malic = y_malic

    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)

    return X, Y