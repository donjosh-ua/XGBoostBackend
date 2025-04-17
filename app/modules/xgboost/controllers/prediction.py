"""
XGBoost prediction controller.
Handles API endpoints for making predictions with XGBoost models.
"""
import os
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core import config
from app.core.logging import xgboost_logger as logger
from app.domain.schemas import (
    PredictRequest, 
    PredictResponse, 
    PredictWithDataRequest, 
    EvaluateRequest, 
    EvaluateResponse
)
from app.modules.xgboost.services.model import predict_with_model, evaluate_model
from app.common.validation import validate_prediction_input


# Create router
router = APIRouter()


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions with an XGBoost model using a data file.
    
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
    Make predictions with an XGBoost model using inline data.
    
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


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate an XGBoost model on a dataset.
    
    Args:
        request (EvaluateRequest): Evaluation request with model path and data path
        
    Returns:
        EvaluateResponse: Evaluation response with metrics
    """
    try:
        # Validate that the files exist
        if not os.path.exists(request.model_path):
            logger.error(f"Model file not found: {request.model_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
            
        if not os.path.exists(request.data_path):
            logger.error(f"Data file not found: {request.data_path}")
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Evaluate the model
        logger.info(f"Evaluating model {request.model_path} on data {request.data_path}")
        metrics = evaluate_model(request.model_path, request.data_path)
        
        return {"metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 