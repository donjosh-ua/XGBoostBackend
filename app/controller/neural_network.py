from pydantic import BaseModel
from app.config import conf_manager
from app.common.common_methods import plot_accuracy_lines_and_curves
from app.domain.models import TrainResponse
from fastapi import APIRouter, HTTPException, Response
import os
from app.service.neural_network_model import (
    train_neural_network,
    predict_with_neural_network,
    save_confusion_matrix,
    compare_neural_networks,
)
from typing import List, Dict, Any, Optional
import torch
import numpy as np

router = APIRouter()


class NeuralNetworkConfig(BaseModel):
    alpha: float = 0.001  # learning rate
    epoch: int = 100  # number of epochs
    criteria: str = "cross_entropy"  # loss function
    optimizer: str = "SGD"  # optimizer algorithm
    hidden_layers: List[int] = []  # sizes of hidden layers
    activation: str = "Tanh"  # activation function
    momentum: float = 0.9  # momentum for SGD
    decay: float = 0.0  # learning rate decay
    batch_size: int = 64  # batch size
    image: bool = False  # whether data is image data
    image_size: Optional[int] = None  # image size for flattening
    Bay: bool = False  # whether to use Bayesian approach
    Lambda: float = 0.005  # regularization parameter


class TrainNNRequest(BaseModel):
    method: str  # "split" or "cv"
    value: int  # if method=="cv", value represents num_folds; for "split", value represents test split percentage
    config: NeuralNetworkConfig  # neural network configuration


class PredictNNRequest(BaseModel):
    data: List[List[float]]  # input data for prediction
    image: bool = False  # whether data is image
    image_size: Optional[int] = None  # size of flattened image if applicable


def get_selected_filepath() -> str:
    """Get the path of the selected data file."""
    selected_file = conf_manager.get_value("selected_file")
    if not selected_file:
        raise HTTPException(
            status_code=400,
            detail="No file has been selected. Please select and load a data file first.",
        )

    loaded_data_path = conf_manager.get_value("loaded_data_path")
    if loaded_data_path and not os.path.isabs(loaded_data_path):
        base_dir = os.path.abspath(os.curdir)
        return os.path.join(base_dir, loaded_data_path)

    return loaded_data_path


@router.post("/train", response_model=TrainResponse)
async def train_nn_model(request: TrainNNRequest):
    """
    Endpoint to train a neural network model using the data file provided in settings.config.
    """
    try:
        data_path = get_selected_filepath()

        # Save training method and configuration in the settings file
        conf_manager.set_value("nn_training_method", request.method)

        # For cross validation, value is number of folds
        # For split, value is percentage of test split (convert from percentage to proportion)
        value = request.value if request.method.lower() == "cv" else request.value / 100
        conf_manager.set_value("nn_training_value", value)

        # Update the configuration with the value
        config_dict = request.config.dict()
        if request.method.lower() == "cv":
            config_dict["cv"] = True
            config_dict["Kfold"] = request.value
        else:
            config_dict["cv"] = False
            config_dict["test_size"] = value

        # Save config to settings
        conf_manager.set_value("nn_config", config_dict)

        # Set a model name based on the configuration
        model_name = f"NN_{request.method}_{request.config.optimizer}"
        config_dict["save_mod"] = model_name

        # Train the network
        result, metrics = train_neural_network(data_path, config_dict, request.method)

        # Save the metrics to settings
        conf_manager.set_value("nn_metrics", metrics)

        return {"message": "Neural Network trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualizations/{plot_type}")
async def get_visualization(plot_type: str):
    """
    Endpoint to get visualization plots generated during neural network training.
    
    Args:
        plot_type (str): Type of plot to retrieve (history, confusion_matrix, comparison)
        
    Returns:
        The requested image file
    """
    try:
        # Get model name from config
        nn_config = conf_manager.get_value("nn_config")
        if not nn_config:
            raise HTTPException(
                status_code=400,
                detail="No neural network configuration found. Please train a model first."
            )
        
        model_name = nn_config.get("save_mod", "NNModel")
        method = conf_manager.get_value("nn_training_method", "split")
        
        # Define plot paths based on type
        plots_folder = "app/data/plots"
        if not os.path.exists(plots_folder):
            raise HTTPException(
                status_code=404,
                detail="Plots directory not found."
            )
        
        plot_path = ""
        if plot_type == "history":
            if method.lower() == "cv":
                plot_path = os.path.join(plots_folder, f"{model_name}_cv_history.png")
            else:
                plot_path = os.path.join(plots_folder, f"{model_name}_history.png")
        elif plot_type == "confusion_matrix":
            plot_path = os.path.join(plots_folder, f"{model_name}_confusion_matrix.png")
        elif plot_type == "comparison":
            plot_path = os.path.join(plots_folder, "nn_comparison.png")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plot type: {plot_type}. Valid types are: history, confusion_matrix, comparison"
            )
        
        # Check if the plot exists
        if not os.path.exists(plot_path):
            raise HTTPException(
                status_code=404,
                detail=f"Plot not found: {plot_path}"
            )
        
        # Read the image file
        with open(plot_path, "rb") as f:
            image_data = f.read()
        
        # Return the image
        return Response(content=image_data, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_with_nn(request: PredictNNRequest):
    """
    Endpoint to make predictions using a trained neural network model.
    """
    try:
        # Get the model path from the settings
        nn_config = conf_manager.get_value("nn_config")
        if not nn_config:
            raise HTTPException(
                status_code=400,
                detail="No neural network configuration found. Please train a model first.",
            )

        model_name = nn_config.get("save_mod", "NNModel")

        # Check if it's a CV model
        method = conf_manager.get_value("nn_training_method", "split")

        # Determine the model path
        if method.lower() == "cv":
            # Use the best CV model (first fold for simplicity)
            model_path = f"app/data/neural_network_models/best_{model_name}_K1"
        else:
            model_path = f"app/data/neural_network_models/best_{model_name}"

        # Check if the model exists
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found at {model_path}. Please train the model first.",
            )

        # Load the model
        model = torch.load(model_path)

        # Convert input data to numpy array
        X = np.array(request.data)

        # Make predictions
        predictions = predict_with_neural_network(
            model, X, image=request.image, image_size=request.image_size
        )

        # Convert numpy array to list for JSON serialization
        predictions_list = (
            predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        )

        return {"predictions": predictions_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=TrainResponse)
async def compare_nn_configurations(request: TrainNNRequest):
    """
    Endpoint to train and compare two different neural network configurations.
    """
    try:
        data_path = get_selected_filepath()

        # Save the base configuration
        base_config = request.config.dict()
        base_config["save_mod"] = "NN_base"

        # Create a variation of the configuration for comparison
        custom_config = request.config.dict()

        # Modify some parameters for the custom configuration
        if custom_config.get("optimizer") == "SGD":
            custom_config["optimizer"] = "Adam"
        else:
            custom_config["optimizer"] = "SGD"

        # Adjust hidden layers
        if not custom_config.get("hidden_layers"):
            # If no hidden layers specified, create a slightly different architecture
            num_features = (
                len(request.data[0])
                if hasattr(request, "data") and request.data
                else 10
            )
            custom_config["hidden_layers"] = [max(int(num_features * 2), 20)]
        else:
            # Modify existing hidden layers
            custom_config["hidden_layers"] = [
                layer * 2 for layer in custom_config["hidden_layers"]
            ]

        custom_config["save_mod"] = "NN_custom"

        # Train both networks
        _, base_metrics = train_neural_network(data_path, base_config, request.method)
        _, custom_metrics = train_neural_network(
            data_path, custom_config, request.method
        )

        # Create comparison plot
        plots_dir = os.path.join("app", "data", "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plot_path = os.path.join(plots_dir, "nn_comparison.png")

        compare_neural_networks(base_metrics, custom_metrics, plot_path)

        return {"message": "Neural Network comparison completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
