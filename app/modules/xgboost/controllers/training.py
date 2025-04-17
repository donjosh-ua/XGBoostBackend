"""
XGBoost training controller.
Handles API endpoints for XGBoost model training.
"""
import os
from fastapi import APIRouter, HTTPException

from app.core import config
from app.core.logging import xgboost_logger as logger
from app.domain.schemas import TrainRequest, TrainResponse
from app.modules.xgboost.services.model import train_normal_xgboost
from app.modules.xgboost.services.bayesian import train_custom_xgboost
from app.common.utils import plot_accuracy_lines_and_curves
from app.common.data_handling import get_dataset_path


# Create router
router = APIRouter()


def get_selected_filepath() -> str:
    """
    Get the path to the selected dataset.
    
    Returns:
        str: Path to the selected dataset
        
    Raises:
        HTTPException: If no file has been selected
    """
    selected_file = config.get_value("selected_file")
    if not selected_file:
        logger.error("No file has been selected")
        raise HTTPException(
            status_code=400,
            detail="No file has been selected. Please select and load a data file first."
        )
    
    try:
        return get_dataset_path(selected_file)
    except FileNotFoundError:
        logger.error(f"Selected file not found: {selected_file}")
        raise HTTPException(
            status_code=404,
            detail=f"Selected file not found: {selected_file}"
        )


@router.post("/normal", response_model=TrainResponse)
async def train_normal_model(request: TrainRequest):
    """
    Train a normal XGBoost model.
    
    Args:
        request (TrainRequest): Training request
        
    Returns:
        TrainResponse: Training response with status message
    """
    try:
        # Get the selected dataset
        data_path = get_selected_filepath()
        
        # Save training method and value in the settings file
        config.set_value("training_method", request.method)
        # For CV, value is number of folds; for split, value is percentage of test split (convert to proportion)
        value = request.value if request.method.lower() == "cv" else request.value / 100
        config.set_value("training_value", value)
        config.set_value("rounds", request.rounds)
        
        # Retrieve model parameters
        model_params = config.get_value("model_parameters", {})
        
        # Update with any provided parameters
        if request.params:
            model_params.update(request.params)
            config.set_value("model_parameters", model_params)
        
        # Train the model
        logger.info(f"Training normal XGBoost model with method={request.method}, value={value}")
        model, evals_result = train_normal_xgboost(data_path, model_params, request.method)
        
        # Extract metrics from the evaluation results
        metrics = {}
        if evals_result and "test" in evals_result:
            for metric in evals_result["test"]:
                if evals_result["test"][metric]:
                    metrics[metric] = evals_result["test"][metric][-1]
        
        logger.info("Normal model training completed successfully")
        return {"message": "Modelo entrenado exitosamente", "metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error training normal model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom", response_model=TrainResponse)
async def train_custom_model(request: TrainRequest):
    """
    Train a custom XGBoost model with a PyMC-adjusted objective function.
    
    Args:
        request (TrainRequest): Training request
        
    Returns:
        TrainResponse: Training response with status message
    """
    try:
        # Get the selected dataset
        data_path = get_selected_filepath()
        
        # Validate distribution
        if not request.distribution:
            logger.error("Distribution is required for custom training")
            raise HTTPException(
                status_code=400,
                detail="Distribution parameter is required for custom training"
            )
        
        # Save parameters in the settings file
        config.set_value("training_method", request.method)
        # For CV, value is number of folds; for split, value is percentage of test split (convert to proportion)
        value = request.value if request.method.lower() == "cv" else request.value / 100
        config.set_value("training_value", value)
        config.set_value("rounds", request.rounds)
        config.set_value("distribution", request.distribution)
        
        # Save custom parameters if provided
        if request.params:
            config.set_value("custom_parameters", request.params)
        
        # Retrieve model parameters
        model_params = config.get_value("model_parameters", {})
        
        # Train the model
        logger.info(f"Training custom XGBoost model with method={request.method}, distribution={request.distribution}")
        model, evals_result = train_custom_xgboost(
            data_path, 
            model_params, 
            request.distribution, 
            request.method
        )
        
        # Extract metrics from the evaluation results
        metrics = {}
        if evals_result and "test" in evals_result:
            for metric in evals_result["test"]:
                if evals_result["test"][metric]:
                    metrics[metric] = evals_result["test"][metric][-1]
        
        logger.info("Custom model training completed successfully")
        return {"message": "Modelo entrenado exitosamente", "metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error training custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/both", response_model=TrainResponse)
async def train_both_models(request: TrainRequest):
    """
    Train both normal and custom XGBoost models and compare them.
    
    Args:
        request (TrainRequest): Training request
        
    Returns:
        TrainResponse: Training response with status message
    """
    try:
        # Get the selected dataset
        data_path = get_selected_filepath()
        
        # Validate distribution
        if not request.distribution:
            logger.error("Distribution is required for custom training")
            raise HTTPException(
                status_code=400,
                detail="Distribution parameter is required for custom training"
            )
        
        # Save parameters in the settings file
        config.set_value("training_method", request.method)
        # For CV, value is number of folds; for split, value is percentage of test split (convert to proportion)
        value = request.value if request.method.lower() == "cv" else request.value / 100
        config.set_value("training_value", value)
        config.set_value("rounds", request.rounds)
        config.set_value("distribution", request.distribution)
        
        # Save custom parameters if provided
        if request.params:
            config.set_value("custom_parameters", request.params)
        
        # Retrieve model parameters
        model_params = config.get_value("model_parameters", {})
        
        # Train normal model
        logger.info("Training normal XGBoost model")
        _, normal_results = train_normal_xgboost(
            data_path, 
            model_params, 
            request.method
        )
        
        # Train custom model
        logger.info(f"Training custom XGBoost model with distribution={request.distribution}")
        _, custom_results = train_custom_xgboost(
            data_path, 
            model_params, 
            request.distribution, 
            request.method
        )
        
        # Create comparison plot
        plots_dir = os.path.join("app", "data", "outputs", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, "accuracies.png")
        
        logger.info("Creating comparison plot")
        plot_accuracy_lines_and_curves(normal_results, custom_results, plot_path)
        
        # Extract metrics from both models
        metrics = {}
        
        # Normal model metrics
        if normal_results and "test" in normal_results:
            for metric in normal_results["test"]:
                if normal_results["test"][metric]:
                    metrics[f"normal_{metric}"] = normal_results["test"][metric][-1]
        
        # Custom model metrics
        if custom_results and "test" in custom_results:
            for metric in custom_results["test"]:
                if custom_results["test"][metric]:
                    metrics[f"custom_{metric}"] = custom_results["test"][metric][-1]
        
        logger.info("Both models trained successfully")
        return {"message": "Ambos modelos entrenados exitosamente", "metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error training both models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 