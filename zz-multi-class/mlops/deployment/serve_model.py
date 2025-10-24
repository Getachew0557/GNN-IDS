"""
Model Serving with FastAPI

This module provides a REST API for serving trained attack graph
classification models in production environments.
"""
"""
Model Serving with FastAPI

This module provides a REST API for serving trained attack graph
classification models in production environments.
"""

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from typing import List, Dict, Any  # Added List import
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[List[float]]  # 2D array: [num_nodes, feature_dim]
    edge_index: List[List[int]]  # 2D array: [2, num_edges]
    model_type: str = "GCN"  # Model type to use for prediction


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predictions: List[int]
    probabilities: List[List[float]]
    confidence: float
    model_used: str


class ModelServer:
    """
    FastAPI server for serving attack graph classification models.
    
    Provides REST endpoints for model inference, health checks,
    and model management.
    """
    
    def __init__(self, models: Dict[str, torch.nn.Module], config: Dict[str, Any]):
        """
        Initialize model server.
        
        Args:
            models: Dictionary of loaded models
            config: Server configuration
        """
        self.app = FastAPI(
            title="Attack Graph Classification API",
            description="REST API for multiclass attack graph classification",
            version="1.0.0"
        )
        self.models = models
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._setup_routes()
        logger.info(f"Model server initialized with {len(models)} models on {self.device}")
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Attack Graph Classification API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "models_loaded": list(self.models.keys()),
                "device": str(self.device)
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint."""
            try:
                return self._make_prediction(request)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            """List available models."""
            return {
                "available_models": list(self.models.keys()),
                "default_model": self.config.get('default_model', 'GCN')
            }
    
    def _make_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make prediction using the specified model.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        # Get model
        model_type = request.model_type
        if model_type not in self.models:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        model = self.models[model_type]
        model.eval()
        model.to(self.device)
        
        # Prepare input data
        features = torch.tensor(request.features, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        edge_index = torch.tensor(request.edge_index, dtype=torch.long)
        
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if hasattr(model, 'name') and model.name == 'NN':
                # Handle NN model (no graph structure)
                rt_meas_dim = getattr(model, 'rt_meas_dim', features.shape[-1])
                action_mask = getattr(model, 'action_mask', list(range(features.shape[1])))
                output = model(features[:, action_mask, -rt_meas_dim:])
            else:
                # Handle graph models
                output = model(features, edge_index)
            
            probabilities = torch.softmax(output, dim=-1)
            predictions = torch.argmax(output, dim=-1)
        
        # Convert to Python types
        predictions = predictions.cpu().numpy().flatten().tolist()
        probabilities = probabilities.cpu().numpy().reshape(-1, probabilities.shape[-1]).tolist()
        
        # Calculate average confidence
        confidence = float(np.max(probabilities, axis=1).mean())
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            model_used=model_type
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the model server.
        
        Args:
            host: Server host
            port: Server port
        """
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def load_models_from_mlflow(experiment_name: str = "attack_graph_classification") -> Dict[str, torch.nn.Module]:
    """
    Load trained models from MLFlow.
    
    Args:
        experiment_name: MLFlow experiment name
        
    Returns:
        Dictionary of model names to loaded models
    """
    import mlflow.pytorch
    
    models = {}
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment {experiment_name} not found")
            return models
        
        # Search for runs with logged models
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        for _, run in runs.iterrows():
            try:
                model_uri = f"runs:/{run.run_id}/model"
                model = mlflow.pytorch.load_model(model_uri)
                
                model_name = run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else f"model_{run.run_id}"
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load model from run {run.run_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error loading models from MLFlow: {e}")
    
    return models


if __name__ == "__main__":
    # Load configuration
    import yaml
    with open("../../mlops/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load models (in production, you might load from model registry)
    models = load_models_from_mlflow()
    
    if not models:
        logger.warning("No models loaded from MLFlow. Using placeholder.")
        # In production, you would load your production model here
        from src.models import GCN
        models = {"GCN": GCN(135, 64, 8)}  # Example dimensions
    
    # Start server
    server = ModelServer(models, config.get('deployment', {}))
    server.run(
        host=config.get('deployment', {}).get('host', '0.0.0.0'),
        port=config.get('deployment', {}).get('api_port', 8000)
    )