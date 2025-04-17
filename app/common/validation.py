"""
Validation utilities for input data and parameters.
"""
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.exceptions import ValidationError
from app.core.logging import app_logger as logger


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(file_path)


def validate_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate XGBoost model parameters and set defaults for missing values.
    
    Args:
        params (Dict[str, Any]): Model parameters
        
    Returns:
        Dict[str, Any]: Validated parameters with defaults for missing values
        
    Raises:
        ValidationError: If required parameters are missing or invalid
    """
    # Default parameters
    default_params = {
        "eta": 0.05,
        "max_depth": 5,
        "gamma": 0.1,
        "learning_rate": 0.1,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 1,
        "seed": 1994,
        "objective": "binary:logistic",
        "eval_metric": "error"
    }
    
    # Validate and merge with defaults
    validated_params = default_params.copy()
    
    # Update with provided parameters
    if params:
        validated_params.update(params)
    
    # Validate numeric parameters
    numeric_params = {
        "eta": (0.0, 1.0),
        "max_depth": (1, 20),
        "gamma": (0.0, 10.0),
        "learning_rate": (0.0, 1.0),
        "min_child_weight": (0, 20),
        "subsample": (0.0, 1.0),
        "colsample_bytree": (0.0, 1.0)
    }
    
    errors = {}
    for param, (min_val, max_val) in numeric_params.items():
        if param in validated_params:
            try:
                value = float(validated_params[param])
                if value < min_val or value > max_val:
                    errors[param] = f"Value must be between {min_val} and {max_val}"
                else:
                    validated_params[param] = value
            except (ValueError, TypeError):
                errors[param] = "Value must be numeric"
    
    # Validate categorical parameters
    if "objective" in validated_params:
        objective = validated_params["objective"]
        valid_objectives = ["binary:logistic", "multi:softmax", "reg:squarederror"]
        if objective not in valid_objectives:
            errors["objective"] = f"Must be one of: {', '.join(valid_objectives)}"
    
    if errors:
        logger.error(f"Parameter validation errors: {errors}")
        raise ValidationError("Invalid model parameters", errors)
    
    return validated_params


def validate_dataset(data_path: str, has_header: bool = False, separator: str = ",") -> Tuple[bool, Optional[str]]:
    """
    Validate a dataset file for correctness.
    
    Args:
        data_path (str): Path to the dataset file
        has_header (bool, optional): Whether the file has a header. Defaults to False.
        separator (str, optional): Column separator. Defaults to ",".
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        # Check file exists
        if not validate_file_exists(data_path):
            return False, "File does not exist"
        
        # Try to read the file
        header = 0 if has_header else None
        df = pd.read_csv(data_path, header=header, sep=separator)
        
        # Check if empty
        if df.empty:
            return False, "Dataset is empty"
        
        # Check for at least one feature and one target column
        if df.shape[1] < 2:
            return False, "Dataset must have at least one feature column and one target column"
            
        # Success
        return True, None
    except Exception as e:
        logger.error(f"Dataset validation error: {e}")
        return False, str(e)


def validate_neural_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate neural network configuration and set defaults for missing values.
    
    Args:
        config (Dict[str, Any]): Neural network configuration
        
    Returns:
        Dict[str, Any]: Validated configuration with defaults for missing values
        
    Raises:
        ValidationError: If required parameters are missing or invalid
    """
    # Default configuration
    default_config = {
        "alpha": 0.001,
        "epoch": 100,
        "criteria": "cross_entropy",
        "optimizer": "SGD",
        "hidden_layers": [],
        "activation": "Tanh",
        "momentum": 0.9,
        "decay": 0.0,
        "batch_size": 64,
        "image": False,
        "Bay": False,
        "Lambda": 0.005
    }
    
    # Validate and merge with defaults
    validated_config = default_config.copy()
    
    # Update with provided configuration
    if config:
        validated_config.update(config)
    
    # Validate numeric parameters
    numeric_params = {
        "alpha": (0.0, 1.0),
        "epoch": (1, 10000),
        "momentum": (0.0, 1.0),
        "decay": (0.0, 1.0),
        "batch_size": (1, 1000),
        "Lambda": (0.0, 1.0)
    }
    
    errors = {}
    for param, (min_val, max_val) in numeric_params.items():
        if param in validated_config:
            try:
                value = float(validated_config[param])
                if value < min_val or value > max_val:
                    errors[param] = f"Value must be between {min_val} and {max_val}"
                else:
                    validated_config[param] = value
            except (ValueError, TypeError):
                errors[param] = "Value must be numeric"
    
    # Validate categorical parameters
    if "criteria" in validated_config:
        criteria = validated_config["criteria"]
        valid_criteria = ["cross_entropy", "mse"]
        if criteria not in valid_criteria:
            errors["criteria"] = f"Must be one of: {', '.join(valid_criteria)}"
    
    if "optimizer" in validated_config:
        optimizer = validated_config["optimizer"]
        valid_optimizers = ["SGD", "Adam", "RMSprop"]
        if optimizer not in valid_optimizers:
            errors["optimizer"] = f"Must be one of: {', '.join(valid_optimizers)}"
    
    if "activation" in validated_config:
        activation = validated_config["activation"]
        valid_activations = ["Tanh", "Sigmoid", "ReLU"]
        if activation not in valid_activations:
            errors["activation"] = f"Must be one of: {', '.join(valid_activations)}"
    
    # Validate hidden_layers is a list of integers
    if "hidden_layers" in validated_config:
        hidden_layers = validated_config["hidden_layers"]
        if not isinstance(hidden_layers, list):
            errors["hidden_layers"] = "Must be a list"
        else:
            try:
                validated_config["hidden_layers"] = [int(x) for x in hidden_layers]
            except (ValueError, TypeError):
                errors["hidden_layers"] = "Must be a list of integers"
    
    if errors:
        logger.error(f"Neural network configuration validation errors: {errors}")
        raise ValidationError("Invalid neural network configuration", errors)
    
    return validated_config


def validate_prediction_input(data: Union[List[List[float]], np.ndarray], expected_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Validate prediction input data.
    
    Args:
        data (Union[List[List[float]], np.ndarray]): Input data for prediction
        expected_shape (Optional[Tuple[int, ...]], optional): Expected shape of the data. Defaults to None.
        
    Returns:
        np.ndarray: Validated data as numpy array
        
    Raises:
        ValidationError: If the data is invalid
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        
        # Check if it's a numpy array
        if not isinstance(data, np.ndarray):
            raise ValidationError("Input data must be a list or numpy array")
        
        # Check if it has the correct number of dimensions
        if len(data.shape) < 2:
            # Add batch dimension if missing
            data = np.expand_dims(data, axis=0)
        
        # Check expected shape if provided
        if expected_shape is not None:
            if len(expected_shape) == 2:  # (n_samples, n_features)
                n_features = expected_shape[1]
                if data.shape[1] != n_features:
                    raise ValidationError(f"Input data must have {n_features} features, but has {data.shape[1]}")
            elif len(expected_shape) > 2:  # More complex shape
                if data.shape[1:] != expected_shape[1:]:
                    raise ValidationError(f"Input data has wrong shape. Expected {expected_shape[1:]}, got {data.shape[1:]}")
        
        return data
    except Exception as e:
        if not isinstance(e, ValidationError):
            logger.error(f"Prediction input validation error: {e}")
            raise ValidationError(f"Invalid prediction input: {e}")
        raise 