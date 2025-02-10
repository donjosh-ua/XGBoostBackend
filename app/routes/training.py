import os
from fastapi import APIRouter, HTTPException
from models.xgboost_model import train_normal_xgboost, train_custom_xgboost
from utils.common_methods import plot_accuracy_lines_and_curves
from schemas.models import TrainResponse
from pydantic import BaseModel
from utils import conf_manager

router = APIRouter()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class TrainRequest(BaseModel):
    method: str  # "split" or "cv"
    value: int   # if method=="cv", value represents num_folds; for "split", a default is used
    rounds: int  # number of rounds for custom training
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
        value = request.value if request.method.lower() == "cv" else request.value/100
        conf_manager.set_value("training_value", value)

        # Retrieve model parameters from the settings file
        model_params = conf_manager.get_value("model_parameters")
        
        _, evals_result = train_normal_xgboost(data_path, model_params, request.method)
        
        print("Model trained successfully")
        return {"message": "Modelo entrenado exitosamente"}
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
        value = request.value if request.method.lower() == "cv" else request.value/100
        conf_manager.set_value("training_value", value)
        conf_manager.set_value("rounds", request.rounds)
        conf_manager.set_value("distribution", request.distribution)
        conf_manager.set_value("custom_parameters", request.params)

        # Retrieve custom parameters from the settings file
        model_params = conf_manager.get_value("model_parameters")
        
        if request.method.lower() != "split":
            raise HTTPException(status_code=400, detail="El entrenamiento custom solo está soportado con el método 'split'.")
        _, evals_result = train_custom_xgboost(data_path, model_params, request.distribution, request.method)
        
        return {"message": "Modelo entrenado exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/both", response_model=TrainResponse)
async def train_both_models(request: TrainRequest):
    """
    Endpoint to train both normal and custom XGBoost models using the data file provided in settings.config.
    """
    try:
        data_path = get_selected_filepath()

        # Save training method and value in the settings file
        conf_manager.set_value("training_method", request.method)
        value = request.value if request.method.lower() == "cv" else request.value/100
        conf_manager.set_value("training_value", value)
        conf_manager.set_value("rounds", request.rounds)
        conf_manager.set_value("distribution", request.distribution)
        conf_manager.set_value("custom_parameters", request.params)

        # Retrieve custom parameters from the settings file
        model_params = conf_manager.get_value("model_parameters")
        
        _, normal_results = train_normal_xgboost(data_path, model_params, request.method)
        _, custom_results = train_custom_xgboost(data_path, model_params, request.distribution, request.method)

        BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        plots_dir = os.path.join(BASE_DIR, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        plot_path = os.path.join(plots_dir, "accuracies.png")

        plot_accuracy_lines_and_curves(normal_results, custom_results, plot_path)
        
        return {"message": "Modelos entrenados exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))