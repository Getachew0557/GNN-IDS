"""
MLFlow Setup and Configuration

This module configures MLFlow for experiment tracking, model registry,
and provides utilities for managing MLFlow experiments.
"""

import mlflow
import mlflow.pytorch
import os
import yaml
from typing import Dict, Any, Optional


class MLFlowManager:
    """
    Manager for MLFlow experiment tracking and model registry.
    
    Provides a unified interface for logging experiments, parameters,
    metrics, and models to MLFlow.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize MLFlow manager with configuration.
        
        Args:
            config_path: Path to MLFlow configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_mlflow()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def _setup_mlflow(self) -> None:
        """Setup MLFlow tracking and registry."""
        # Set tracking URI
        tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'mlruns/')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create directories if they don't exist
        os.makedirs(tracking_uri, exist_ok=True)
        
        # Set experiment
        experiment_name = self.config.get('mlflow', {}).get('experiment_name', 'attack_graph_classification')
        mlflow.set_experiment(experiment_name)
        
        print(f"MLFlow configured: tracking_uri={tracking_uri}, experiment={experiment_name}")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Name of the run
            tags: Optional tags for the run
            
        Returns:
            Active MLFlow run
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log PyTorch model to current run."""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(self, local_path: str) -> None:
        """Log artifact to current run."""
        mlflow.log_artifact(local_path)
    
    def register_model(self, model_uri: str, model_name: str) -> None:
        """
        Register model in MLFlow model registry.
        
        Args:
            model_uri: URI of the logged model
            model_name: Name for the registered model
        """
        try:
            mlflow.register_model(model_uri, model_name)
            print(f"Model {model_name} registered successfully")
        except Exception as e:
            print(f"Failed to register model: {e}")
    
    def get_best_run(self, experiment_id: str, metric: str = "val_accuracy") -> Dict[str, Any]:
        """
        Retrieve the best run from an experiment based on a metric.
        
        Args:
            experiment_id: MLFlow experiment ID
            metric: Metric to optimize
            
        Returns:
            Dictionary with best run information
        """
        try:
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            if runs.empty:
                return {}
            
            best_run = runs.loc[runs[f"metrics.{metric}"].idxmax()]
            return best_run.to_dict()
        except Exception as e:
            print(f"Error retrieving best run: {e}")
            return {}


def setup_mlflow_experiment(experiment_name: str = "attack_graph_classification") -> MLFlowManager:
    """
    Convenience function to setup MLFlow experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured MLFlow manager
    """
    config = {
        'mlflow': {
            'tracking_uri': 'mlruns/',
            'experiment_name': experiment_name
        }
    }
    
    # Write temporary config
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    manager = MLFlowManager('temp_config.yaml')
    os.remove('temp_config.yaml')
    
    return manager