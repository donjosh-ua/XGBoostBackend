"""
Neural network prediction controller.
Handles API endpoints for making predictions with neural network models.
"""
import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.core.logging import nn_logger as logger
from app.domain.schemas import PredictRequest, PredictResponse, PredictWithDataRequest
from app.modules.neural_network.services.model import predict_with_model, calculate_metrics
from app.common.validation import validate_prediction_input

# Create router
router = APIRouter()


class BayesianPredictRequest(BaseModel):
    """Request for Bayesian prediction with uncertainty."""
    model_path: str
    data_path: str
    num_samples: int = 100


class BayesianPredictResponse(BaseModel):
    """Response with Bayesian predictions including uncertainty."""
    predictions: List[float]
    uncertainties: Optional[List[float]] = None
    metrics: Optional[dict] = None


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions with a neural network model using a data file.
    
    Args:
        request (PredictRequest): Prediction request with model path and data path
        
    Returns:
        PredictResponse: Prediction response with predictions
    """
    try:
        # Validate that the files exist
        if not os.path.exists(request.model_path):
            logger.error(f"Model file not found: {request.model_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
            
        if not os.path.exists(request.data_path):
            logger.error(f"Data file not found: {request.data_path}")
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Make predictions
        logger.info(f"Making predictions with model {request.model_path} on data {request.data_path}")
        predictions = predict_with_model(request.model_path, request.data_path)
        
        # Convert predictions to list
        predictions_list = predictions.tolist()
        
        return {"predictions": predictions_list}
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/with-data", response_model=PredictResponse)
async def predict_with_data(request: PredictWithDataRequest):
    """
    Make predictions with a neural network model using inline data.
    
    Args:
        request (PredictWithDataRequest): Prediction request with model path and input data
        
    Returns:
        PredictResponse: Prediction response with predictions
    """
    try:
        # Validate that the model file exists
        if not os.path.exists(request.model_path):
            logger.error(f"Model file not found: {request.model_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        # Validate input data
        try:
            data = validate_prediction_input(request.data)
        except ValueError as e:
            logger.error(f"Invalid input data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
        
        # Make predictions
        logger.info(f"Making predictions with model {request.model_path} on inline data")
        predictions = predict_with_model(request.model_path, data)
        
        # Convert predictions to list
        predictions_list = predictions.tolist()
        
        return {"predictions": predictions_list}
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bayesian", response_model=BayesianPredictResponse)
async def bayesian_predict(request: BayesianPredictRequest):
    """
    Make Bayesian predictions with uncertainty estimation.
    
    Args:
        request (BayesianPredictRequest): Prediction request
        
    Returns:
        BayesianPredictResponse: Prediction response with predictions and uncertainties
    """
    try:
        # Validate that the files exist
        if not os.path.exists(request.model_path):
            logger.error(f"Model file not found: {request.model_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
            
        if not os.path.exists(request.data_path):
            logger.error(f"Data file not found: {request.data_path}")
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # For Bayesian prediction, we'd typically use multiple prediction samples and calculate statistics
        import torch
        import pyro
        
        # Load the data
        df = pd.read_csv(request.data_path)
        X = df.iloc[:, :-1].values
        if df.shape[1] > 1:  # If there are enough columns, the last is the target
            y_true = df.iloc[:, -1].values
        else:
            y_true = None
            
        # Load the model
        loaded = torch.load(request.model_path)
        
        # Check if it's a Bayesian model (tuple of model and guide)
        if not isinstance(loaded, tuple):
            logger.error("The provided model is not a Bayesian model")
            raise HTTPException(status_code=400, detail="The provided model is not a Bayesian model")
            
        model, guide = loaded
        
        # Setup for prediction
        pyro.clear_param_store()
        X_tensor = torch.FloatTensor(X)
        
        # Make multiple predictions for uncertainty estimation
        predictive = pyro.infer.Predictive(model, guide=guide, num_samples=request.num_samples)
        samples = predictive(X_tensor)["obs"]
        
        # Calculate mean and standard deviation for each prediction
        means = samples.mean(0).numpy().flatten()
        stds = samples.std(0).numpy().flatten()
        
        # Calculate metrics if true values are available
        metrics = None
        if y_true is not None:
            metrics = calculate_metrics(y_true, means)
        
        return {
            "predictions": means.tolist(),
            "uncertainties": stds.tolist(),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error making Bayesian predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 