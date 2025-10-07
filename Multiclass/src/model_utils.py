import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def add_adversarial_noise(X, epsilon=0.2):  # Increased epsilon for more noise
    noise = torch.normal(0, epsilon, size=X.shape).to(X.device)
    X_adv = X + noise
    return X_adv.clamp(0, 1)

def add_label_noise(y, noise_rate=0.05, num_classes=8):
    """Add random label noise to a fraction of the labels."""
    y_noisy = y.clone()
    batch_size, num_action_nodes = y.shape
    num_noisy = int(noise_rate * batch_size * num_action_nodes)
    idx = torch.randperm(batch_size * num_action_nodes)[:num_noisy]
    noisy_labels = torch.randint(0, num_classes, (num_noisy,), device=y.device)
    y_noisy.view(-1)[idx] = noisy_labels
    return y_noisy

def update_edge_index(edge_index, num_nodes, max_edges=100):
    if edge_index.shape[1] >= max_edges:
        return edge_index
    num_new_edges = min(2, max_edges - edge_index.shape[1])
    new_edges = torch.randint(0, num_nodes, (2, num_new_edges), dtype=torch.long)
    new_edges = new_edges[:, new_edges[0] != new_edges[1]]
    if new_edges.shape[1] > 0:
        edge_index = torch.cat([edge_index, new_edges], dim=1)
    return edge_index[:, :max_edges]

def train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim, device='cpu', patience=15):
    # Class weights: higher for malicious classes to balance benign (35000) vs. malicious (8000-12000)
    class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # Increased weight_decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    action_mask = model.action_mask
    dynamic_edge_index = edge_index.clone().to(device)
    if model.name == 'NN':
        X_train = X_train[:, action_mask, -rt_meas_dim:].clone()
        X_val = X_val[:, action_mask, -rt_meas_dim:].clone()
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Reduced batch_size

    stat = {'loss_train': [], 'loss_val': [], 'acc_train': [], 'acc_val': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if (epoch + 1) % 5 == 0 and model.name != 'NN':
            dynamic_edge_index = update_edge_index(dynamic_edge_index, num_nodes=X_train.shape[1], max_edges=100)
            if model.name == 'GCN-EW':
                model.update_edge_weight(dynamic_edge_index.shape[1])
            print(f'Epoch {epoch + 1}: Updated edge_index to {dynamic_edge_index.shape[1]} edges')

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y_noisy = add_label_noise(batch_y, noise_rate=0.05)  # Add 5% label noise
            if model.name == 'NN':
                output = model(batch_X)
                output = output.permute(0, 2, 1)  # [batch_size, num_action_nodes, out_dim] -> [batch_size, out_dim, num_action_nodes]
                loss = criterion(output, batch_y_noisy)
            else:
                output = model(batch_X, dynamic_edge_index)
                output = output[:, action_mask].permute(0, 2, 1)
                loss = criterion(output, batch_y_noisy)
            batch_X_adv = add_adversarial_noise(batch_X)
            if model.name == 'NN':
                output_adv = model(batch_X_adv)
                output_adv = output_adv.permute(0, 2, 1)
                loss_adv = criterion(output_adv, batch_y_noisy)
            else:
                output_adv = model(batch_X_adv, dynamic_edge_index)
                output_adv = output_adv[:, action_mask].permute(0, 2, 1)
                loss_adv = criterion(output_adv, batch_y_noisy)
            total_loss = 0.7 * loss + 0.3 * loss_adv
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_loss.item()

        loss_train, acc_train = evaluate_loss_acc(model, X_train, Y_train, criterion, dynamic_edge_index, device)
        loss_val, acc_val = evaluate_loss_acc(model, X_val, Y_val, criterion, dynamic_edge_index, device)

        stat['loss_train'].append(loss_train)
        stat['loss_val'].append(loss_val)
        stat['acc_train'].append(acc_train)
        stat['acc_val'].append(acc_val)

        # Manual logging for learning rate changes
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss_val)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f'Epoch {epoch + 1}: Learning rate reduced to {new_lr:.6f}')

        print('Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, LR: {:.6f}'.format(
            epoch + 1, loss_train, acc_train, loss_val, acc_val, new_lr))
        
        if loss_val < best_val_loss - 0.001:  # Stricter improvement check
            best_val_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
    model.stat = stat
    model.dynamic_edge_index = dynamic_edge_index
    return model

def evaluate_loss_acc(model, X, y, criterion, edge_index, device='cpu', batch_size=128):
    model.eval()
    mask = model.action_mask
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            edge_index = edge_index.to(device)
            
            if model.name == 'NN':
                output = model(batch_X)
                output = output.permute(0, 2, 1)  # [batch_size, num_action_nodes, out_dim] -> [batch_size, out_dim, num_action_nodes]
                loss = criterion(output, batch_y)
                y_pred = torch.argmax(output, dim=1)
            else:
                output = model(batch_X, edge_index)
                output = output[:, mask].permute(0, 2, 1)
                loss = criterion(output, batch_y)
                y_pred = torch.argmax(output, dim=1)
            
            total_loss += loss.item() * batch_X.size(0)
            total_correct += (y_pred == batch_y).sum().item()
            total_samples += batch_y.size(0) * batch_y.size(1)
    
    avg_loss = total_loss / len(dataset)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def predict_prob(model, X, edge_index, rt_meas_dim, device='cpu'):
    model.eval()
    mask = model.action_mask
    prob = torch.zeros((len(X), len(mask), 8), dtype=torch.float32, device=device)  # 8 classes
    X = X.to(device)
    edge_index = edge_index.to(device)
    rt_meas_dim = int(rt_meas_dim)
    with torch.no_grad():
        if model.name == 'NN':
            X_sliced = X[:, mask, -rt_meas_dim:]
            prob = torch.softmax(model(X_sliced), dim=-1)
        else:
            prob = torch.softmax(model(X, edge_index)[:, mask], dim=-1)
    return prob

def evaluate_performance(models, X, y, edge_index, device='cpu'):
    metrics = []
    for name, model in models.items():
        model.eval()
        prob = predict_prob(model, X, getattr(model, 'dynamic_edge_index', edge_index), rt_meas_dim=model.rt_meas_dim, device=device)
        pred_ts = torch.argmax(prob, dim=2)
        accuracy = (pred_ts == y).sum().item() / (y.shape[0] * y.shape[1])
        conf_matrix = confusion_matrix(y.flatten().cpu(), pred_ts.flatten().cpu(), labels=range(8))
        precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), pred_ts.view(-1).cpu(), average='macro', labels=range(8))
        d = {
            'model': name,
            'conf_matrix': conf_matrix.tolist(),
            'precision': '{:.4f}'.format(precision),
            'recall': '{:.4f}'.format(recall),
            'f1': '{:.4f}'.format(f1),
            'loss_train': '{:.4f}'.format(model.stat['loss_train'][-1]),
            'loss_val': '{:.4f}'.format(model.stat['loss_val'][-1]),
            'acc_train': '{:.4f}'.format(model.stat['acc_train'][-1]),
            'acc_val': '{:.4f}'.format(model.stat['acc_val'][-1]),
            'accuracy': '{:.4f}'.format(accuracy)
        }
        metrics.append(d)
    
    top_models = ['GCN-EW', 'GAT', 'GraphSAGE', 'TAD-GAT']
    if all(m in models for m in top_models):
        ensemble_probs = torch.zeros_like(predict_prob(models[top_models[0]], X, getattr(models[top_models[0]], 'dynamic_edge_index', edge_index), rt_meas_dim=models[top_models[0]].rt_meas_dim, device=device))
        for m_name in top_models:
            ensemble_probs += predict_prob(models[m_name], X, getattr(models[m_name], 'dynamic_edge_index', edge_index), rt_meas_dim=models[m_name].rt_meas_dim, device=device)
        ensemble_probs /= len(top_models)
        ensemble_pred = torch.argmax(ensemble_probs, dim=2)
        ensemble_accuracy = (ensemble_pred == y).sum().item() / (y.shape[0] * y.shape[1])
        conf_matrix = confusion_matrix(y.flatten().cpu(), ensemble_pred.flatten().cpu(), labels=range(8))
        precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), ensemble_pred.view(-1).cpu(), average='macro', labels=range(8))
        d_ens = {
            'model': 'Ensemble',
            'conf_matrix': conf_matrix.tolist(),
            'precision': '{:.4f}'.format(precision),
            'recall': '{:.4f}'.format(recall),
            'f1': '{:.4f}'.format(f1),
            'loss_train': '{:.4f}'.format(models['GAT'].stat['loss_train'][-1]),
            'loss_val': '{:.4f}'.format(models['GAT'].stat['loss_val'][-1]),
            'acc_train': '{:.4f}'.format(models['GAT'].stat['acc_train'][-1]),
            'acc_val': '{:.4f}'.format(models['GAT'].stat['acc_val'][-1]),
            'accuracy': '{:.4f}'.format(ensemble_accuracy)
        }
        metrics.append(d_ens)
    
    return metrics