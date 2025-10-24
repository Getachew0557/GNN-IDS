"""
Evaluation Module for Attack Graph Models

This module provides comprehensive evaluation utilities including
performance metrics, visualization, and model comparison.
"""

"""
Evaluation Module for Attack Graph Models

This module provides comprehensive evaluation utilities including
performance metrics, visualization, and model comparison.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional  # Added List import
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelEvaluator:
    """
    Comprehensive model evaluator for multiclass classification.
    
    Provides detailed evaluation metrics, visualizations, and model comparisons
    for attack graph classification tasks.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names or [
            'Benign', 'Web Attack - Brute Force', 'DoS slowloris', 
            'FTP-Patator', 'SSH-Patator', 'DDoS', 'Bot', 'PortScan'
        ]
    
    def evaluate_model(self, model: torch.nn.Module, X_test: torch.Tensor, 
                      Y_test: torch.Tensor, edge_index: torch.Tensor,
                      rt_meas_dim: int, device: str = 'cuda') -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            Y_test: Test labels
            edge_index: Graph connectivity
            rt_meas_dim: Runtime measurement dimension
            device: Evaluation device
            
        Returns:
            Dictionary of evaluation results
        """
        # Move data to device
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        edge_index = edge_index.to(device)
        model.to(device)
        
        # Get predictions
        with torch.no_grad():
            if hasattr(model, 'name') and model.name == 'NN':
                output = model(X_test[:, model.action_mask, -rt_meas_dim:])
            else:
                output = model(X_test, edge_index)
            
            probabilities = torch.softmax(output, dim=-1)
            predictions = torch.argmax(output, dim=-1)
        
        # Flatten for metric calculation
        Y_true = Y_test.cpu().numpy().flatten()
        Y_pred = predictions.cpu().numpy().flatten()
        Y_prob = probabilities.cpu().numpy().reshape(-1, probabilities.shape[-1])
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(Y_true, Y_pred, Y_prob)
        
        # Create confusion matrix
        cm = confusion_matrix(Y_true, Y_pred, labels=range(len(self.class_names)))
        
        results = {
            'model_name': getattr(model, 'name', 'Unknown'),
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': predictions.cpu().numpy(),
            'probabilities': Y_prob,
            'true_labels': Y_true
        }
        
        return results
    
    def _calculate_comprehensive_metrics(self, Y_true: np.ndarray, 
                                       Y_pred: np.ndarray, 
                                       Y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            Y_true: True labels
            Y_pred: Predicted labels
            Y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(Y_true, Y_pred)
        precision = precision_score(Y_true, Y_pred, average='weighted', zero_division=0)
        recall = recall_score(Y_true, Y_pred, average='weighted', zero_division=0)
        f1 = f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(Y_true, Y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(Y_true, Y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        
        # Additional metrics
        cm = confusion_matrix(Y_true, Y_pred, labels=range(len(self.class_names)))
        specificity_per_class = []
        for i in range(len(self.class_names)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        metrics = {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'per_class': {
                'precision': dict(zip(self.class_names, precision_per_class)),
                'recall': dict(zip(self.class_names, recall_per_class)),
                'f1_score': dict(zip(self.class_names, f1_per_class)),
                'specificity': dict(zip(self.class_names, specificity_per_class))
            }
        }
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            model_name: str = "Model") -> go.Figure:
        """
        Create interactive confusion matrix plot.
        
        Args:
            confusion_matrix: Confusion matrix array
            model_name: Name of the model for title
            
        Returns:
            Plotly figure object
        """
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=self.class_names,
            y=self.class_names,
            title=f"{model_name} - Confusion Matrix",
            color_continuous_scale='Blues'
        )
        
        # Add annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(confusion_matrix[i, j]),
                    showarrow=False,
                    font=dict(color='red' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black')
                )
        
        fig.update_layout(width=600, height=600)
        return fig
    
    def plot_metrics_comparison(self, results_list: List[Dict[str, Any]]) -> go.Figure:
        """
        Create comparison plot for multiple models.
        
        Args:
            results_list: List of evaluation results
            
        Returns:
            Plotly figure with model comparisons
        """
        model_names = [result['model_name'] for result in results_list]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[metric.capitalize() for metric in metrics],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [result['metrics']['overall'][metric] for result in results_list]
            
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric.capitalize()),
                row=row, col=col
            )
            
            fig.update_yaxes(range=[0, 1], row=row, col=col)
        
        fig.update_layout(
            height=600,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def generate_classification_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate detailed classification report.
        
        Args:
            results: Evaluation results
            
        Returns:
            DataFrame with per-class metrics
        """
        metrics = results['metrics']['per_class']
        
        report_data = []
        for class_name in self.class_names:
            report_data.append({
                'Class': class_name,
                'Precision': metrics['precision'][class_name],
                'Recall': metrics['recall'][class_name],
                'F1-Score': metrics['f1_score'][class_name],
                'Specificity': metrics['specificity'][class_name]
            })
        
        # Add overall metrics
        overall_metrics = results['metrics']['overall']
        report_data.append({
            'Class': 'OVERALL',
            'Precision': overall_metrics['precision'],
            'Recall': overall_metrics['recall'],
            'F1-Score': overall_metrics['f1_score'],
            'Specificity': 'N/A'
        })
        
        return pd.DataFrame(report_data)


def evaluate_performance(models: Dict[str, torch.nn.Module], 
                        X_test: torch.Tensor, Y_test: torch.Tensor,
                        edge_index: torch.Tensor, device: str = 'cuda') -> List[Dict[str, Any]]:
    """
    Evaluate multiple models and return comprehensive results.
    
    Args:
        models: Dictionary of model names to models
        X_test: Test features
        Y_test: Test labels
        edge_index: Graph connectivity
        device: Evaluation device
        
    Returns:
        List of evaluation results for each model
    """
    evaluator = ModelEvaluator()
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Ensure model has required attributes
        if not hasattr(model, 'rt_meas_dim'):
            model.rt_meas_dim = X_test.shape[-1]  # Default to full dimension
        
        result = evaluator.evaluate_model(
            model, X_test, Y_test, edge_index, model.rt_meas_dim, device
        )
        results.append(result)
        
        # Print summary
        metrics = result['metrics']['overall']
        print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
              f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results


def predict_prob(model: torch.nn.Module, X: torch.Tensor, 
                edge_index: torch.Tensor, rt_meas_dim: int, 
                device: str = 'cuda') -> torch.Tensor:
    """
    Generate probability predictions from model.
    
    Args:
        model: Trained model
        X: Input features
        edge_index: Graph connectivity
        rt_meas_dim: Runtime measurement dimension
        device: Inference device
        
    Returns:
        Probability tensor
    """
    model.eval()
    model.to(device)
    X = X.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'name') and model.name == 'NN':
            output = model(X[:, model.action_mask, -rt_meas_dim:])
        else:
            output = model(X, edge_index)
        probabilities = torch.softmax(output, dim=-1)
    
    return probabilities.cpu()