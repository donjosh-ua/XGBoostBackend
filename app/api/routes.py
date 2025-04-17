"""
API routes configuration.
Registers all API routes from different modules.
"""
from fastapi import APIRouter

# Import module routers
from app.modules.xgboost.controllers.training import router as xgboost_training_router
from app.modules.xgboost.controllers.prediction import router as xgboost_prediction_router
from app.modules.xgboost.controllers.tuning import router as xgboost_tuning_router
from app.modules.xgboost.controllers.testing import router as xgboost_testing_router
# Neural network module not fully implemented yet
# from app.modules.neural_network.controllers.training import router as nn_training_router
# from app.modules.neural_network.controllers.prediction import router as nn_prediction_router
from app.modules.data_management.controllers.data_file import router as data_router


def create_router() -> APIRouter:
    """
    Create and configure the main API router.
    
    Returns:
        APIRouter: Configured router with all endpoints
    """
    # Create main router
    main_router = APIRouter()
    
    # Register module routers with prefixes and tags
    main_router.include_router(data_router, prefix="/data", tags=["Data Management"])
    
    # XGBoost routers
    main_router.include_router(xgboost_training_router, prefix="/xgboost/train", tags=["XGBoost Training"])
    main_router.include_router(xgboost_prediction_router, prefix="/xgboost/predict", tags=["XGBoost Prediction"])
    main_router.include_router(xgboost_tuning_router, prefix="/xgboost/parameters", tags=["XGBoost Parameters"])
    main_router.include_router(xgboost_testing_router, prefix="/xgboost/test", tags=["XGBoost Testing"])
    
    # Neural Network routers - commented out until implementation is complete
    # main_router.include_router(nn_training_router, prefix="/neural-network/train", tags=["Neural Network Training"])
    # main_router.include_router(nn_prediction_router, prefix="/neural-network/predict", tags=["Neural Network Prediction"])
    
    return main_router


# Create the main router
router = create_router() 