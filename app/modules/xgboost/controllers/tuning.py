"""
XGBoost tuning controller.
Handles API endpoints for tuning XGBoost model parameters.
"""
import os
from fastapi import APIRouter, HTTPException

from app.core import config
from app.core.logging import xgboost_logger as logger
from app.domain.schemas import TuningRequest, TuningResponse, MessageResponse
from app.modules.xgboost.services.model import grid_search_xgboost
from app.common.validation import validate_model_parameters


# Create router
router = APIRouter()


@router.post("/grid-search", response_model=TuningResponse)
async def perform_grid_search(request: TuningRequest):
    """
    Perform grid search to find optimal XGBoost parameters.
    
    Args:
        request (TuningRequest): Tuning request
        
    Returns:
        TuningResponse: Tuning response with best parameters
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
        
        # Perform grid search
        logger.info("Performing grid search for XGBoost parameters")
        best_params = grid_search_xgboost()
        
        # Save the best parameters to the configuration
        model_params = config.get_value("model_parameters", {})
        model_params.update(best_params)
        config.set_value("model_parameters", model_params)
        
        # Calculate best score (placeholder - in a real implementation, this would be returned by the grid search)
        best_score = 0.95  # Placeholder value
        
        return {
            "message": "Búsqueda de parámetros completada exitosamente",
            "best_params": best_params,
            "best_score": best_score
        }
    
    except Exception as e:
        logger.error(f"Error performing grid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update", response_model=MessageResponse)
async def update_parameters(params: dict):
    """
    Update XGBoost model parameters.
    
    Args:
        params (dict): XGBoost parameters to update
        
    Returns:
        MessageResponse: Success message
    """
    try:
        # Validate the parameters
        validated_params = validate_model_parameters(params)
        
        # Get current parameters
        model_params = config.get_value("model_parameters", {})
        
        # Update with new parameters
        model_params.update(validated_params)
        
        # Save the updated parameters
        config.set_value("model_parameters", model_params)
        
        logger.info(f"Updated XGBoost parameters: {validated_params}")
        
        return {"message": "Parámetros actualizados exitosamente"}
    
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current", response_model=dict)
async def get_current_parameters():
    """
    Get current XGBoost model parameters.
    
    Returns:
        dict: Current parameters
    """
    try:
        model_params = config.get_value("model_parameters", {})
        
        # If no parameters have been set, use default values
        if not model_params:
            model_params = {
                "eta": 0.05,
                "max_depth": 5,
                "gamma": 0.1,
                "learning_rate": 0.1,
                "min_child_weight": 3,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "seed": 1994,
                "objective": "binary:logistic",
                "eval_metric": "error"
            }
            
        return model_params
    
    except Exception as e:
        logger.error(f"Error getting current parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/distribution", response_model=MessageResponse)
async def update_distribution_parameters(params: dict):
    """
    Update parameters for the custom distribution used in the PyMC-adjusted objective function.
    
    Args:
        params (dict): Distribution parameters to update
        
    Returns:
        MessageResponse: Success message
    """
    try:
        # Get current distribution parameters
        custom_params = config.get_value("custom_parameters", {})
        
        # Update with new parameters
        custom_params.update(params)
        
        # Save the updated parameters
        config.set_value("custom_parameters", custom_params)
        
        logger.info(f"Updated distribution parameters: {params}")
        
        return {"message": "Parámetros de distribución actualizados exitosamente"}
    
    except Exception as e:
        logger.error(f"Error updating distribution parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 