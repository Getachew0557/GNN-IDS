"""
Training utilities for attack graph classification models
Includes training loops, early stopping, and model saving
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
"""
Training Module for Attack Graph Models

This module provides comprehensive training utilities with MLOps integration,
including experiment tracking, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Dict, List, Any, Optional  # Added List import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLFlow not available. Install with: pip install mlflow")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            
        return self.early_stop
    
    def restore_best_weights(self, model: nn.Module):
        """Restore model to best weights"""
        if self.restore_best and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info("Restored model weights from best validation loss")


class ModelTrainer:
    """
    Comprehensive model trainer with logging, checkpointing, and evaluation
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training statistics
        self.statistics = {
            'loss_train': [],
            'loss_val': [],
            'acc_train': [],
            'acc_val': [],
            'precision_train': [],
            'precision_val': [],
            'recall_train': [],
            'recall_val': [],
            'f1_train': [],
            'f1_val': [],
            'epoch_times': []
        }
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: nn.Module,
              num_epochs: int,
              patience: int = 10,
              checkpoint_dir: Optional[str] = None,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, List[float]]:
        """
        Train the model with comprehensive logging and evaluation
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of training epochs
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
            scheduler: Learning rate scheduler
            
        Returns:
            Training statistics
        """
        early_stopping = EarlyStopping(patience=patience)
        
        logger.info(f"Starting training for {self.model.name} on {self.device}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            epoch_time = time.time() - start_time
            
            # Update statistics
            for key in train_metrics.keys():
                self.statistics[f'{key}_train'].append(train_metrics[key])
                self.statistics[f'{key}_val'].append(val_metrics[key])
            self.statistics['epoch_times'].append(epoch_time)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Log progress
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            
            # Check early stopping
            if early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                early_stopping.restore_best_weights(self.model)
                break
            
            # Save checkpoint
            if checkpoint_dir and epoch % 10 == 0:
                self._save_checkpoint(epoch, optimizer, checkpoint_dir)
        
        logger.info(f"Training completed for {self.model.name}")
        return self.statistics
    
    def _train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> Dict[str, float]:
        """Single training epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            preds = torch.argmax(output, dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _validate_epoch(self, val_loader: torch.utils.data.DataLoader,
                       criterion: nn.Module) -> Dict[str, float]:
        """Single validation epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
                
                total_loss += loss.item()
                preds = torch.argmax(output, dim=-1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch progress"""
        if epoch % 10 == 0:
            logger.info(
                f'Epoch {epoch:03d} | Time: {epoch_time:.2f}s | '
                f'Train Loss: {train_metrics["loss"]:.4f} | Val Loss: {val_metrics["loss"]:.4f} | '
                f'Train Acc: {train_metrics["accuracy"]:.4f} | Val Acc: {val_metrics["accuracy"]:.4f}'
            )
    
    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, checkpoint_dir: str):
        """Save training checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'statistics': self.statistics,
            'model_name': self.model.name
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{self.model.name}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save best model
        best_model_path = os.path.join(checkpoint_dir, f'best_model_{self.model.name}.pth')
        torch.save(self.model.state_dict(), best_model_path)


def create_data_loaders(X_train: torch.Tensor, Y_train: torch.Tensor,
                       X_val: torch.Tensor, Y_val: torch.Tensor,
                       X_test: torch.Tensor, Y_test: torch.Tensor,
                       batch_size: int = 32) -> Tuple:
    """
    Create PyTorch data loaders for training, validation, and test sets
    
    Args:
        X_train, Y_train: Training data and labels
        X_val, Y_val: Validation data and labels
        X_test, Y_test: Test data and labels
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def setup_training(model: nn.Module, learning_rate: float = 0.001, 
                  weight_decay: float = 1e-4) -> Tuple:
    """
    Setup training components: optimizer, loss function, and scheduler
    
    Args:
        model: Model to train
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        Tuple of (optimizer, criterion, scheduler)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    return optimizer, criterion, scheduler