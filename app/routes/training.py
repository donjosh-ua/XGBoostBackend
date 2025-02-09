import os
from fastapi import APIRouter, HTTPException
from models.xgboost_model import train_normal_xgboost, train_custom_xgboost
from schemas.models import TrainResponse
from pydantic import BaseModel
from utils import conf_manager

router = APIRouter()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class TrainRequest(BaseModel):
    method: str  # "split" or "cv"
    value: int   # if method=="cv", value represents num_folds; for "split", a default is used
    distribution: str = None  # used for custom training
    params: dict  # additional parameters if needed

def get_selected_filepath() -> str:
    selected_file = conf_manager.get_value("selected_file")
    if not selected_file:
        raise HTTPException(status_code=400, detail="No file has been selected. Please select and load a data file first.")
    
    # Use the loaded_data_path from settings if available, otherwise use DATA_DIR + selected_file
    loaded_data_path = conf_manager.get_value("loaded_data_path")
    if loaded_data_path and os.path.isfile(loaded_data_path):
        return loaded_data_path
    
    return os.path.join(DATA_DIR, selected_file)

@router.post("/normal", response_model=TrainResponse)
async def train_normal_model(request: TrainRequest):
    """
    Endpoint to train a normal XGBoost model using the data file provided in settings.config.
    """
    try:
        data_path = get_selected_filepath()
        # Save training method and value in the settings file
        conf_manager.set_value("training_method", request.method)
        conf_manager.set_value("training_value", request.value)
        
        num_folds = request.value if request.method.lower() == "cv" else 5
        model, evals_result = train_normal_xgboost(data_path, request.params, request.method, num_folds=num_folds)
        
        return {"message": "Modelo entrenado exitosamente", "evals_result": evals_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom", response_model=TrainResponse)
async def train_custom_model(request: TrainRequest):
    """
    Endpoint to train a custom XGBoost model using the data file provided in settings.config.
    Only the "split" method is supported.
    """
    try:
        data_path = get_selected_filepath()
        # Save training method and value in the settings file
        conf_manager.set_value("training_method", request.method)
        conf_manager.set_value("training_value", request.value)
        
        distribution = request.params.get("distribution", "Normal")
        if request.method.lower() != "split":
            raise HTTPException(status_code=400, detail="El entrenamiento custom solo está soportado con el método 'split'.")
        model, evals_result = train_custom_xgboost(data_path, request.params, distribution, request.method)
        
        return {"message": "Modelo entrenado exitosamente", "evals_result": evals_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))