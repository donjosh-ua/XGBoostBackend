"""
XGBoost testing controller.
Handles API endpoints for testing XGBoost models.
"""
import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core import config
from app.core.logging import xgboost_logger as logger
from app.domain.schemas import TrainRequest, EvaluateResponse, MessageResponse
from app.modules.xgboost.services.model import cross_validate
from app.common.utils import save_confusion_matrix
from app.common.data_handling import load_data_from_csv


# Create router
router = APIRouter()


@router.post("/cross-validate", response_model=EvaluateResponse)
async def perform_cross_validation(request: TrainRequest):
    """
    Perform cross-validation with the current parameters.
    
    Args:
        request (TrainRequest): Training request with method and value
        
    Returns:
        EvaluateResponse: Evaluation metrics from cross-validation
    """
    try:
        # Check if a data file is selected
        selected_file = config.get_value("selected_file")
        if not selected_file:
            logger.error("No file has been selected")
            raise HTTPException(
                status_code=400,
                detail="No file has been selected. Please select and load a data file first."
            )
        
        # Load data
        datafile = config.get_value("loaded_data_path")
        header = config.get_value("has_header", False)
        header = 0 if header else None
        separator = config.get_value("separator", ",")
        
        # Read the dataset
        data = pd.read_csv(datafile, header=header, sep=separator)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Get model parameters
        model_params = config.get_value("model_parameters", {})
        
        # Determine if it's a multiclass problem
        num_classes = len(np.unique(y))
        is_multiclass = num_classes > 2
        
        # Perform cross-validation
        logger.info(f"Performing {request.value}-fold cross-validation")
        metrics = cross_validate(
            X=X,
            y=y,
            params=model_params,
            n_splits=request.value,
            is_multiclass=is_multiclass
        )
        
        return {"metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error performing cross-validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confusion-matrix", response_model=MessageResponse)
async def generate_confusion_matrix():
    """
    Generate and save a confusion matrix visualization for the current model.
    
    Returns:
        MessageResponse: Success message with path to the saved confusion matrix
    """
    try:
        # Check if a data file is selected
        selected_file = config.get_value("selected_file")
        if not selected_file:
            logger.error("No file has been selected")
            raise HTTPException(
                status_code=400,
                detail="No file has been selected. Please select and load a data file first."
            )
        
        # Check if models exist
        normal_model_path = os.path.join("app", "data", "models", "xgboost", "model_normal.xgb")
        custom_model_path = os.path.join("app", "data", "models", "xgboost", "model_custom.xgb")
        
        if not os.path.exists(normal_model_path) and not os.path.exists(custom_model_path):
            logger.error("No trained models found")
            raise HTTPException(
                status_code=400,
                detail="No trained models found. Please train a model first."
            )
        
        # Load the data
        _, _, _, _, test_x, test_y = load_data_from_csv()
        
        # Ensure output directory exists
        output_dir = os.path.join("app", "data", "outputs", "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Choose which model to use (prefer normal model if available)
        model_path = normal_model_path if os.path.exists(normal_model_path) else custom_model_path
        model_type = "normal" if model_path == normal_model_path else "custom"
        
        # Make predictions
        import xgboost as xgb
        model = xgb.Booster(model_file=model_path)
        dtest = xgb.DMatrix(test_x)
        y_pred_prob = model.predict(dtest)
        
        # Convert probabilities to class labels
        if len(y_pred_prob.shape) > 1:  # multiclass
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:  # binary
            y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Generate confusion matrix
        output_path = os.path.join(output_dir, f"confusion_matrix_{model_type}.png")
        save_confusion_matrix(test_y, y_pred, output_path)
        
        logger.info(f"Confusion matrix saved to {output_path}")
        
        return {"message": f"Confusion matrix saved to {output_path}"}
    
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-models", response_model=EvaluateResponse)
async def compare_models():
    """
    Compare the performance of normal and custom models.
    
    Returns:
        EvaluateResponse: Comparative metrics
    """
    try:
        # Check if models exist
        normal_model_path = os.path.join("app", "data", "models", "xgboost", "model_normal.xgb")
        custom_model_path = os.path.join("app", "data", "models", "xgboost", "model_custom.xgb")
        
        if not os.path.exists(normal_model_path):
            logger.error("Normal model not found")
            raise HTTPException(
                status_code=400,
                detail="Normal model not found. Please train a normal model first."
            )
            
        if not os.path.exists(custom_model_path):
            logger.error("Custom model not found")
            raise HTTPException(
                status_code=400,
                detail="Custom model not found. Please train a custom model first."
            )
        
        # Load test data
        _, _, _, _, test_x, test_y = load_data_from_csv()
        
        # Make predictions with both models
        import xgboost as xgb
        
        normal_model = xgb.Booster(model_file=normal_model_path)
        custom_model = xgb.Booster(model_file=custom_model_path)
        
        dtest = xgb.DMatrix(test_x)
        
        normal_pred_prob = normal_model.predict(dtest)
        custom_pred_prob = custom_model.predict(dtest)
        
        # Convert probabilities to class labels
        if len(normal_pred_prob.shape) > 1:  # multiclass
            normal_pred = np.argmax(normal_pred_prob, axis=1)
            custom_pred = np.argmax(custom_pred_prob, axis=1)
        else:  # binary
            normal_pred = (normal_pred_prob >= 0.5).astype(int)
            custom_pred = (custom_pred_prob >= 0.5).astype(int)
        
        # Calculate metrics for both models
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        normal_metrics = {
            "accuracy": float(accuracy_score(test_y, normal_pred)),
            "precision": float(precision_score(test_y, normal_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(test_y, normal_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(test_y, normal_pred, average="weighted", zero_division=0))
        }
        
        custom_metrics = {
            "accuracy": float(accuracy_score(test_y, custom_pred)),
            "precision": float(precision_score(test_y, custom_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(test_y, custom_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(test_y, custom_pred, average="weighted", zero_division=0))
        }
        
        # Combine metrics
        metrics = {
            "normal": normal_metrics,
            "custom": custom_metrics,
            "improvement": {
                metric: float(custom_metrics[metric] - normal_metrics[metric])
                for metric in normal_metrics
            }
        }
        
        logger.info("Model comparison completed")
        
        return {"metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 