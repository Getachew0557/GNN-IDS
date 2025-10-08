import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def add_adversarial_noise(X, epsilon=0.05):
    noise = torch.normal(0, epsilon, size=X.shape).to(X.device)
    X_adv = X + noise
    return X_adv.clamp(0, 1)

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
    num_class_0 = (Y_train == 0).sum().item()
    num_class_1 = (Y_train == 1).sum().item()
    pos_weight = torch.tensor([num_class_0 / num_class_1 * 1.1], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    action_mask = model.action_mask
    dynamic_edge_index = edge_index.clone().to(device)
    if model.name == 'NN':
        X_train = X_train[:, action_mask, -rt_meas_dim:].clone()
        X_val = X_val[:, action_mask, -rt_meas_dim:].clone()
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

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
            if model.name == 'NN':
                output = model(batch_X)
                loss = criterion(output, batch_y)
            else:
                output = model(batch_X, dynamic_edge_index)
                loss = criterion(output[:, action_mask], batch_y)
            batch_X_adv = add_adversarial_noise(batch_X)
            if model.name == 'NN':
                output_adv = model(batch_X_adv)
                loss_adv = criterion(output_adv, batch_y)
            else:
                output_adv = model(batch_X_adv, dynamic_edge_index)
                loss_adv = criterion(output_adv[:, action_mask], batch_y)
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

        scheduler.step(loss_val)
        
        print('Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, LR: {:.6f}'.format(
            epoch + 1, loss_train, acc_train, loss_val, acc_val, optimizer.param_groups[0]['lr']))
        
        if loss_val < best_val_loss:
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
                loss = criterion(output, batch_y)
                y_pred = torch.sigmoid(output) > 0.5
            else:
                output = model(batch_X, edge_index)
                loss = criterion(output[:, mask], batch_y)
                y_pred = torch.sigmoid(output[:, mask]) > 0.5
            
            total_loss += loss.item() * batch_X.size(0)
            total_correct += (y_pred == batch_y).sum().item()
            total_samples += batch_y.size(0) * batch_y.size(1)
    
    avg_loss = total_loss / len(dataset)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def predict_prob(model, X, edge_index, rt_meas_dim, device='cpu'):
    model.eval()
    mask = model.action_mask
    prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)
    X = X.to(device)
    edge_index = edge_index.to(device)
    rt_meas_dim = int(rt_meas_dim)  # Ensure rt_meas_dim is an integer
    with torch.no_grad():
        if model.name == 'NN':
            X_sliced = X[:, mask, -rt_meas_dim:]  # Slice to action nodes
            prob_1 = torch.sigmoid(model(X_sliced))
            prob = torch.stack([1 - prob_1, prob_1], dim=2)
        else:
            prob_1 = torch.sigmoid(model(X, edge_index))[:, mask]
            prob = torch.stack([1 - prob_1, prob_1], dim=2)
    return prob

def evaluate_performance(models, X, y, edge_index, device='cpu'):
    metrics = []
    for name, model in models.items():
        model.eval()
        prob = predict_prob(model, X, getattr(model, 'dynamic_edge_index', edge_index), rt_meas_dim=model.rt_meas_dim, device=device)
        pred_ts = torch.argmax(prob, dim=2)
        accuracy = (pred_ts == y).sum().item() / (y.shape[0] * y.shape[1])
        conf_matrix = confusion_matrix(y.flatten().cpu(), pred_ts.flatten().cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), pred_ts.view(-1).cpu(), average='macro')
        TN, FP, FN, TP = conf_matrix.ravel()
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        y_probs = prob.view(-1, 2)
        fpr, tpr, _ = roc_curve(y.view(-1).cpu(), y_probs[:, 1].cpu())
        roc_auc = auc(fpr, tpr)
        d = {'model': name, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
             'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
             'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
             'loss_train': '{:.4f}'.format(model.stat['loss_train'][-1]), 'loss_val': '{:.4f}'.format(model.stat['loss_val'][-1]),
             'acc_train': '{:.4f}'.format(model.stat['acc_train'][-1]), 'acc_val': '{:.4f}'.format(model.stat['acc_val'][-1]),
             'accuracy': '{:.4f}'.format(accuracy)}
        metrics.append(d)
    
    top_models = ['GCN-EW', 'GAT', 'GraphSAGE', 'TAD-GAT']
    if all(m in models for m in top_models):
        ensemble_probs = torch.zeros_like(predict_prob(models[top_models[0]], X, getattr(models[top_models[0]], 'dynamic_edge_index', edge_index), rt_meas_dim=models[top_models[0]].rt_meas_dim, device=device))
        for m_name in top_models:
            ensemble_probs += predict_prob(models[m_name], X, getattr(models[m_name], 'dynamic_edge_index', edge_index), rt_meas_dim=models[m_name].rt_meas_dim, device=device)
        ensemble_probs /= len(top_models)
        ensemble_pred = torch.argmax(ensemble_probs, dim=2)
        ensemble_accuracy = (ensemble_pred == y).sum().item() / (y.shape[0] * y.shape[1])
        conf_matrix = confusion_matrix(y.flatten().cpu(), ensemble_pred.flatten().cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), ensemble_pred.view(-1).cpu(), average='macro')
        TN, FP, FN, TP = conf_matrix.ravel()
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        y_probs = ensemble_probs.view(-1, 2)
        fpr, tpr, _ = roc_curve(y.view(-1).cpu(), y_probs[:, 1].cpu())
        roc_auc = auc(fpr, tpr)
        d_ens = {'model': 'Ensemble', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                 'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
                 'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
                 'loss_train': '{:.4f}'.format(models['GAT'].stat['loss_train'][-1]), 'loss_val': '{:.4f}'.format(models['GAT'].stat['loss_val'][-1]),
                 'acc_train': '{:.4f}'.format(models['GAT'].stat['acc_train'][-1]), 'acc_val': '{:.4f}'.format(models['GAT'].stat['acc_val'][-1]),
                 'accuracy': '{:.4f}'.format(ensemble_accuracy)}
        metrics.append(d_ens)
    
    return metrics



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# def add_adversarial_noise(X, epsilon=0.05):
#     noise = torch.normal(0, epsilon, size=X.shape).to(X.device)
#     X_adv = X + noise
#     return X_adv.clamp(0, 1)

# def update_edge_index(edge_index, num_nodes, max_edges=100):
#     if edge_index.shape[1] >= max_edges:
#         return edge_index
#     num_new_edges = min(2, max_edges - edge_index.shape[1])
#     new_edges = torch.randint(0, num_nodes, (2, num_new_edges), dtype=torch.long)
#     new_edges = new_edges[:, new_edges[0] != new_edges[1]]
#     if new_edges.shape[1] > 0:
#         edge_index = torch.cat([edge_index, new_edges], dim=1)
#     return edge_index[:, :max_edges]

# def train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim=234, device='cpu', patience=15):
#     num_class_0 = (Y_train == 0).sum().item()
#     num_class_1 = (Y_train == 1).sum().item()
#     pos_weight = torch.tensor([num_class_0 / num_class_1 * 1.1], dtype=torch.float32).to(device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#     action_mask = model.action_mask
#     dynamic_edge_index = edge_index.clone().to(device)
#     if model.name == 'NN':
#         X_train = X_train[:, action_mask, -rt_meas_dim:].clone()
#         X_val = X_val[:, action_mask, -rt_meas_dim:].clone()
#     dataset = TensorDataset(X_train, Y_train)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

#     stat = {'loss_train': [], 'loss_val': [], 'acc_train': [], 'acc_val': []}
#     best_val_loss = float('inf')
#     epochs_no_improve = 0

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         if (epoch + 1) % 5 == 0 and model.name != 'NN':
#             dynamic_edge_index = update_edge_index(dynamic_edge_index, num_nodes=X_train.shape[1], max_edges=100)
#             if model.name == 'GCN-EW':
#                 model.update_edge_weight(dynamic_edge_index.shape[1])
#             # No update_edge_attention for GraphSAGE, as it doesn't use attention
#             print(f'Epoch {epoch + 1}: Updated edge_index to {dynamic_edge_index.shape[1]} edges')

#         for batch_X, batch_y in dataloader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             if model.name == 'NN':
#                 output = model(batch_X)
#                 loss = criterion(output, batch_y)
#             else:
#                 output = model(batch_X, dynamic_edge_index)
#                 loss = criterion(output[:, action_mask], batch_y)
#             batch_X_adv = add_adversarial_noise(batch_X)
#             if model.name == 'NN':
#                 output_adv = model(batch_X_adv)
#                 loss_adv = criterion(output_adv, batch_y)
#             else:
#                 output_adv = model(batch_X_adv, dynamic_edge_index)
#                 loss_adv = criterion(output_adv[:, action_mask], batch_y)
#             total_loss = 0.7 * loss + 0.3 * loss_adv
#             optimizer.zero_grad()
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += total_loss.item()

#         loss_train, acc_train = evaluate_loss_acc(model, X_train, Y_train, criterion, dynamic_edge_index, device)
#         loss_val, acc_val = evaluate_loss_acc(model, X_val, Y_val, criterion, dynamic_edge_index, device)

#         stat['loss_train'].append(loss_train)
#         stat['loss_val'].append(loss_val)
#         stat['acc_train'].append(acc_train)
#         stat['acc_val'].append(acc_val)

#         scheduler.step(loss_val)
        
#         print('Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, LR: {:.6f}'.format(
#             epoch + 1, loss_train, acc_train, loss_val, acc_val, optimizer.param_groups[0]['lr']))
        
#         if loss_val < best_val_loss:
#             best_val_loss = loss_val
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(f'Early stopping at epoch {epoch + 1}')
#                 break
    
#     model.stat = stat
#     model.dynamic_edge_index = dynamic_edge_index
#     return model

# def evaluate_loss_acc(model, X, y, criterion, edge_index, device='cpu', batch_size=128):
#     model.eval()
#     mask = model.action_mask
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
    
#     dataset = TensorDataset(X, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     with torch.no_grad():
#         for batch_X, batch_y in dataloader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             edge_index = edge_index.to(device)
            
#             if model.name == 'NN':
#                 output = model(batch_X)
#                 loss = criterion(output, batch_y)
#                 y_pred = torch.sigmoid(output) > 0.5
#             else:
#                 output = model(batch_X, edge_index)
#                 loss = criterion(output[:, mask], batch_y)
#                 y_pred = torch.sigmoid(output[:, mask]) > 0.5
            
#             total_loss += loss.item() * batch_X.size(0)
#             total_correct += (y_pred == batch_y).sum().item()
#             total_samples += batch_y.size(0) * batch_y.size(1)
    
#     avg_loss = total_loss / len(dataset)
#     avg_acc = total_correct / total_samples
#     return avg_loss, avg_acc

# def predict_prob(model, X, edge_index, rt_meas_dim=234, device='cpu'):
#     model.eval()
#     mask = model.action_mask
#     prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)
#     X = X.to(device)
#     edge_index = edge_index.to(device)
#     rt_meas_dim = int(rt_meas_dim)  # Ensure rt_meas_dim is an integer
#     with torch.no_grad():
#         if model.name == 'NN':
#             X_sliced = X[:, mask, -rt_meas_dim:]  # Slice to action nodes
#             prob_1 = torch.sigmoid(model(X_sliced))
#             prob = torch.stack([1 - prob_1, prob_1], dim=2)
#         else:
#             prob_1 = torch.sigmoid(model(X, edge_index))[:, mask]
#             prob = torch.stack([1 - prob_1, prob_1], dim=2)
#     return prob

# def evaluate_performance(models, X, y, edge_index, device='cpu'):
#     metrics = []
#     for name, model in models.items():
#         model.eval()
#         prob = predict_prob(model, X, getattr(model, 'dynamic_edge_index', edge_index), rt_meas_dim=model.rt_meas_dim, device=device)
#         pred_ts = torch.argmax(prob, dim=2)
#         accuracy = (pred_ts == y).sum().item() / (y.shape[0] * y.shape[1])
#         conf_matrix = confusion_matrix(y.flatten().cpu(), pred_ts.flatten().cpu())
#         precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), pred_ts.view(-1).cpu(), average='macro')
#         TN, FP, FN, TP = conf_matrix.ravel()
#         FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#         FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
#         y_probs = prob.view(-1, 2)
#         fpr, tpr, _ = roc_curve(y.view(-1).cpu(), y_probs[:, 1].cpu())
#         roc_auc = auc(fpr, tpr)
#         d = {'model': name, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
#              'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
#              'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
#              'loss_train': '{:.4f}'.format(model.stat['loss_train'][-1]), 'loss_val': '{:.4f}'.format(model.stat['loss_val'][-1]),
#              'acc_train': '{:.4f}'.format(model.stat['acc_train'][-1]), 'acc_val': '{:.4f}'.format(model.stat['acc_val'][-1]),
#              'accuracy': '{:.4f}'.format(accuracy)}
#         metrics.append(d)
    
#     top_models = ['GCN-EW', 'GAT', 'GraphSAGE']
#     if all(m in models for m in top_models):
#         ensemble_probs = torch.zeros_like(predict_prob(models[top_models[0]], X, getattr(models[top_models[0]], 'dynamic_edge_index', edge_index), rt_meas_dim=models[top_models[0]].rt_meas_dim, device=device))
#         for m_name in top_models:
#             ensemble_probs += predict_prob(models[m_name], X, getattr(models[m_name], 'dynamic_edge_index', edge_index), rt_meas_dim=models[m_name].rt_meas_dim, device=device)
#         ensemble_probs /= len(top_models)
#         ensemble_pred = torch.argmax(ensemble_probs, dim=2)
#         ensemble_accuracy = (ensemble_pred == y).sum().item() / (y.shape[0] * y.shape[1])
#         conf_matrix = confusion_matrix(y.flatten().cpu(), ensemble_pred.flatten().cpu())
#         precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1).cpu(), ensemble_pred.view(-1).cpu(), average='macro')
#         TN, FP, FN, TP = conf_matrix.ravel()
#         FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#         FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
#         y_probs = ensemble_probs.view(-1, 2)
#         fpr, tpr, _ = roc_curve(y.view(-1).cpu(), y_probs[:, 1].cpu())
#         roc_auc = auc(fpr, tpr)
#         d_ens = {'model': 'Ensemble', 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
#                  'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
#                  'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
#                  'loss_train': '{:.4f}'.format(models['GAT'].stat['loss_train'][-1]), 'loss_val': '{:.4f}'.format(models['GAT'].stat['loss_val'][-1]),
#                  'acc_train': '{:.4f}'.format(models['GAT'].stat['acc_train'][-1]), 'acc_val': '{:.4f}'.format(models['GAT'].stat['acc_val'][-1]),
#                  'accuracy': '{:.4f}'.format(ensemble_accuracy)}
#         metrics.append(d_ens)
    
#     return metrics


