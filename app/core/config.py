"""
Configuration management for the application.
Handles loading, saving, and retrieving configuration values from settings file.
"""
import json
import os
from typing import Any, Dict, Optional, Union

# Configuration file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "settings.config")

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "selected_file": None,
    "training_method": None,
    "training_value": 0.7,
    "kseed": 1994,
    "model_parameters": {
        "eta": 0.05,
        "max_depth": 5,
        "gamma": 0.1,
        "learning_rate": 0.1,
        "min_child_weight": 3,
        "subsample": 0.7,
        "colsample_bytree": 1,
        "seed": 1994,
        "objective": "binary:logistic",
        "scale_pos_weight": 3,
        "eval_metric": "error"
    },
    "rounds": 5,
    "distribution": "Normal",
    "custom_parameters": {
        "mean": 0,
        "sigma": 10,
        "alpha": 0,
        "beta": 0,
        "lambda": 0
    },
    "has_header": False,
    "separator": ","
}


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file. If file doesn't exist, create it with default values.
    
    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If the file is corrupted or doesn't exist, reset to defaults
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config_data: Dict[str, Any]) -> None:
    """
    Save configuration to file.
    
    Args:
        config_data (Dict[str, Any]): Configuration dictionary to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)


def get_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by key.
    
    Args:
        key (str): Configuration key
        default (Any, optional): Default value if key doesn't exist. Defaults to None.
    
    Returns:
        Any: The configuration value or default if not found
    """
    config_data = load_config()
    return config_data.get(key, default)


def set_value(key: str, value: Any) -> None:
    """
    Set a configuration value by key.
    
    Args:
        key (str): Configuration key
        value (Any): Value to set
    """
    config_data = load_config()
    config_data[key] = value
    save_config(config_data)


def update_nested_value(path: str, value: Any) -> None:
    """
    Update a nested configuration value using dot notation.
    
    Args:
        path (str): Path to the value using dot notation (e.g., "model_parameters.eta")
        value (Any): Value to set
    """
    config_data = load_config()
    keys = path.split(".")
    
    # Navigate to the nested dictionary
    current = config_data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    save_config(config_data)


def get_nested_value(path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        path (str): Path to the value using dot notation (e.g., "model_parameters.eta")
        default (Any, optional): Default value if path doesn't exist. Defaults to None.
    
    Returns:
        Any: The configuration value or default if not found
    """
    config_data = load_config()
    keys = path.split(".")
    
    # Navigate to the nested dictionary
    current = config_data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current 