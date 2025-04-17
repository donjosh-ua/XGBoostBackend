"""
Common data handling utilities.
Provides functions for loading, processing, and validating data.
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Union, Any

from app.core import config
from app.core.exceptions import DataError, FileNotFoundError
from app.core.logging import data_logger as logger


def ensure_data_dir_exists() -> None:
    """
    Ensure that the data directory exists.
    """
    data_dir = os.path.join("app", "data", "datasets")
    os.makedirs(data_dir, exist_ok=True)


def get_available_datasets() -> List[str]:
    """
    Get a list of available CSV datasets.
    
    Returns:
        List[str]: List of CSV filenames
    """
    data_dir = os.path.join("app", "data", "datasets")
    ensure_data_dir_exists()
    
    try:
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith(".csv")]
        return csv_files
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return []


def get_dataset_path(filename: str) -> str:
    """
    Get the full path to a dataset file.
    
    Args:
        filename (str): Name of the dataset file
    
    Returns:
        str: Full path to the dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("app", "data", "datasets", safe_filename)
    
    if not os.path.isfile(file_path):
        logger.error(f"Dataset file not found: {safe_filename}")
        raise FileNotFoundError(file_path)
    
    return file_path


def load_dataset_preview(filename: str, num_rows: int = 10, has_header: bool = False, 
                         separator: str = ",") -> List[Dict]:
    """
    Load a preview of a dataset.
    
    Args:
        filename (str): Name of the dataset file
        num_rows (int, optional): Number of rows to preview. Defaults to 10.
        has_header (bool, optional): Whether the file has a header row. Defaults to False.
        separator (str, optional): Column separator. Defaults to ",".
    
    Returns:
        List[Dict]: Preview data as a list of dictionaries
        
    Raises:
        DataError: If there's an error loading the data
    """
    try:
        file_path = get_dataset_path(filename)
        header = 0 if has_header else None
        
        df = pd.read_csv(file_path, nrows=num_rows, header=header, sep=separator)
        
        # If no header, create column names
        if header is None:
            df.columns = [f"column_{i}" for i in range(df.shape[1])]
        
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error loading dataset preview: {e}")
        raise DataError(f"Error loading dataset preview: {e}")


def load_data_from_csv() -> Tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from the CSV file specified in the settings and split it into training and test sets.
    
    Returns:
        Tuple containing:
            dtrain (xgb.DMatrix): XGBoost DMatrix for training
            dtest (xgb.DMatrix): XGBoost DMatrix for testing
            train_x (np.ndarray): Training features
            train_y (np.ndarray): Training labels
            test_x (np.ndarray): Test features
            test_y (np.ndarray): Test labels
            
    Raises:
        DataError: If there's an error loading or processing the data
    """
    from sklearn.model_selection import train_test_split
    
    datafile = config.get_value("loaded_data_path")
    if not datafile:
        logger.error("No data file loaded")
        raise DataError("No data file loaded. Please load a data file first.")

    # Load configuration values
    train_ratio = config.get_value("training_value")
    if train_ratio is None or train_ratio >= 1:
        train_ratio = 0.7  # default training ratio

    header = config.get_value("has_header")
    header = 0 if header else None

    seed = config.get_value("kseed", 1994)  # default random seed
    separator = config.get_value("separator", ",")

    try:
        data = pd.read_csv(datafile, header=header, sep=separator)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=(1 - train_ratio), random_state=seed
        )

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        return dtrain, dtest, train_x, train_y, test_x, test_y
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
        raise DataError(f"Error loading data from CSV: {e}")


def get_number_of_classes() -> int:
    """
    Get the number of unique classes in the loaded dataset.
    
    Returns:
        int: Number of unique classes
        
    Raises:
        DataError: If there's an error loading or processing the data
    """
    try:
        datafile = config.get_value("loaded_data_path")
        if not datafile:
            logger.error("No data file loaded")
            raise DataError("No data file loaded. Please load a data file first.")
        
        header = config.get_value("has_header")
        header = 0 if header else None
        separator = config.get_value("separator", ",")
        
        data = pd.read_csv(datafile, header=header, sep=separator)
        
        # Get unique values in the last column (target)
        y = data.iloc[:, -1]
        return len(np.unique(y))
    except Exception as e:
        logger.error(f"Error determining number of classes: {e}")
        raise DataError(f"Error determining number of classes: {e}")


def save_model(model: Any, filename: str, module: str = "xgboost") -> str:
    """
    Save a model to the appropriate directory.
    
    Args:
        model (Any): The model to save
        filename (str): Name for the saved model
        module (str, optional): Module name (xgboost or neural_network). Defaults to "xgboost".
    
    Returns:
        str: Path to the saved model
        
    Raises:
        DataError: If there's an error saving the model
    """
    try:
        # Ensure directory exists
        model_dir = os.path.join("app", "data", "models", module)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create full path
        model_path = os.path.join(model_dir, filename)
        
        # Save the model (method depends on model type)
        if module == "xgboost":
            model.save_model(model_path)
        else:
            # Assuming torch model for neural_network
            import torch
            torch.save(model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise DataError(f"Error saving model: {e}")


def load_model(filename: str, module: str = "xgboost") -> Any:
    """
    Load a model from the appropriate directory.
    
    Args:
        filename (str): Name of the model file
        module (str, optional): Module name (xgboost or neural_network). Defaults to "xgboost".
    
    Returns:
        Any: The loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        DataError: If there's an error loading the model
    """
    try:
        # Create full path
        model_dir = os.path.join("app", "data", "models", module)
        model_path = os.path.join(model_dir, filename)
        
        # Check if file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        
        # Load the model
        if module == "xgboost":
            import xgboost as xgb
            model = xgb.Booster(model_file=model_path)
        else:
            # Assuming torch model for neural_network
            import torch
            model = torch.load(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise DataError(f"Error loading model: {e}") 