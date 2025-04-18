"""
Neural network training controller.
Handles API endpoints for neural network model training.
"""
import os
from fastapi import APIRouter, HTTPException

from app.core import config
from app.core.logging import nn_logger as logger
from app.domain.schemas import TrainResponse, NeuralNetworkConfig
from app.modules.neural_network.services.model import train_neural_network, evaluate_model
from app.common.data_handling import get_dataset_path

# Create router
router = APIRouter()


@router.post("/simple", response_model=TrainResponse)
async def train_simple_model(config_data: NeuralNetworkConfig):
    """
    Train a simple neural network model.
    
    Args:
        config_data (NeuralNetworkConfig): Neural network configuration
        
    Returns:
        TrainResponse: Training response with status message
    """
    try:
        # Get the selected dataset
        selected_file = config.get_value("selected_file")
        if not selected_file:
            logger.error("No file has been selected")
            raise HTTPException(
                status_code=400,
                detail="No file has been selected. Please select and load a data file first."
            )
        
        data_path = get_dataset_path(selected_file)
        
        # Generate a model name for visualization
        model_name = f"simple_nn_{config_data.optimizer}"
        
        # Convert Pydantic model to dict
        nn_config = {
            "hidden_size": config_data.hidden_layers[0] if config_data.hidden_layers else 64,
            "learning_rate": config_data.alpha,
            "epochs": config_data.epoch,
            "batch_size": config_data.batch_size,
            "criteria": config_data.criteria,
            "optimizer": config_data.optimizer,
            "activation": config_data.activation,
            "momentum": config_data.momentum,
            "has_header": config.get_value("has_header", False),
            "separator": config.get_value("separator", ","),
            "bayesian": False,  # Standard neural network
            "save_mod": model_name  # Add the model name for visualization
        }
        
        # Save config to application settings
        config.set_value("nn_config", nn_config)
        config.set_value("nn_training_method", "split")
        
        # Train the model
        logger.info("Training simple neural network model")
        _, metrics = train_neural_network(data_path, nn_config)
        
        # Extract final metrics
        final_metrics = {}
        if "loss" in metrics and metrics["loss"]:
            final_metrics["loss"] = metrics["loss"][-1]
        
        # Add other metrics
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in metrics:
                final_metrics[metric] = metrics[metric]
        
        return {"message": "Neural network model trained successfully", "metrics": final_metrics}
    
    except Exception as e:
        logger.error(f"Error training neural network model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bayesian", response_model=TrainResponse)
async def train_bayesian_model(config_data: NeuralNetworkConfig):
    """
    Train a Bayesian neural network model.
    
    Args:
        config_data (NeuralNetworkConfig): Neural network configuration
        
    Returns:
        TrainResponse: Training response with status message
    """
    try:
        # Get the selected dataset
        selected_file = config.get_value("selected_file")
        if not selected_file:
            logger.error("No file has been selected")
            raise HTTPException(
                status_code=400,
                detail="No file has been selected. Please select and load a data file first."
            )
        
        data_path = get_dataset_path(selected_file)
        
        # Generate a model name for visualization
        model_name = f"bayesian_nn_{config_data.alpha}"
        
        # Convert Pydantic model to dict
        nn_config = {
            "hidden_size": config_data.hidden_layers[0] if config_data.hidden_layers else 64,
            "learning_rate": config_data.alpha,
            "epochs": config_data.epoch,
            "batch_size": config_data.batch_size,
            "criteria": config_data.criteria,
            "activation": config_data.activation,
            "has_header": config.get_value("has_header", False),
            "separator": config.get_value("separator", ","),
            "bayesian": True,  # Bayesian neural network
            "Lambda": config_data.Lambda,  # Regularization parameter for Bayesian network
            "save_mod": model_name  # Add the model name for visualization
        }
        
        # Save config to application settings
        config.set_value("nn_config", nn_config)
        config.set_value("nn_training_method", "split")
        
        # Train the model
        logger.info("Training Bayesian neural network model")
        _, metrics = train_neural_network(data_path, nn_config)
        
        # Extract final metrics
        final_metrics = {}
        if "loss" in metrics and metrics["loss"]:
            final_metrics["loss"] = metrics["loss"][-1]
        
        # Add other metrics
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in metrics:
                final_metrics[metric] = metrics[metric]
        
        return {"message": "Bayesian neural network model trained successfully", "metrics": final_metrics}
    
    except Exception as e:
        logger.error(f"Error training Bayesian neural network model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=TrainResponse)
async def evaluate_nn_model(data_path: str, model_path: str):
    """
    Evaluate a neural network model on a dataset.
    
    Args:
        data_path (str): Path to the evaluation data
        model_path (str): Path to the model
        
    Returns:
        TrainResponse: Evaluation response with metrics
    """
    try:
        # Validate the files exist
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")
            
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        
        # Evaluate the model
        logger.info(f"Evaluating neural network model {model_path} on {data_path}")
        metrics = evaluate_model(model_path, data_path)
        
        return {"message": "Neural network model evaluated successfully", "metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error evaluating neural network model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 