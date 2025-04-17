"""
File management service.
Handles operations related to data files.
"""
import os
import pandas as pd
import aiofiles
from typing import Dict, List, Tuple, Any

from app.core import config
from app.common.data_handling import get_number_of_classes
from app.common.utils import sanitize_filename
from app.core.logging import data_logger as logger


# Constants
DATA_DIR = os.path.join("app", "data", "datasets")


def get_data_files() -> List[str]:
    """
    Get a list of available data files.
    
    Returns:
        List[str]: List of CSV files
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        files = os.listdir(DATA_DIR)
        csv_files = [f for f in files if f.endswith(".csv")]
        return csv_files
    except Exception as e:
        logger.error(f"Error listing data files: {e}")
        raise


def select_file(filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Select a data file for use in training and prediction.
    
    Args:
        filename (str): Name of the file to select
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (message, preview data)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    safe_filename = sanitize_filename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)
    
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {safe_filename}")
        raise FileNotFoundError(file_path)

    # Update config file
    config.set_value("selected_file", safe_filename)

    # Generate preview
    preview = []
    try:
        with open(file_path, "r") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                preview.append(line.rstrip("\n"))
        
        # Convert preview lines to dictionaries
        preview_dicts = []
        for i, line in enumerate(preview):
            preview_dicts.append({"line": i + 1, "content": line})
        
        return f"Selected file set to {safe_filename}", preview_dicts
    except Exception as e:
        logger.error(f"Error generating preview for {safe_filename}: {e}")
        raise


def load_file(filename: str, has_header: bool, separator: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load a data file with specific options.
    
    Args:
        filename (str): Name of the file to load
        has_header (bool): Whether the file has a header
        separator (str): Column separator
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (message, preview data)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    safe_filename = sanitize_filename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)
    
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {safe_filename}")
        raise FileNotFoundError(file_path)

    # Save the selected file path in the config file
    config.set_value("selected_file", safe_filename)

    header = 0 if has_header else None
    try:
        df = pd.read_csv(file_path, sep=separator, header=header)
        
        # Persist additional settings: file path, header info, and separator used.
        config.set_value("loaded_data_path", file_path)
        config.set_value("has_header", has_header)
        config.set_value("separator", separator)

        # Retrieve the loaded data file path from settings
        num_classes = get_number_of_classes()

        # Copy initial parameters from the settings
        params = config.get_value("model_parameters", {})

        # Update parameters based on the number of classes with property adjustments
        if num_classes > 2:
            # Remove scale_pos_weight if exists
            if "scale_pos_weight" in params:
                del params["scale_pos_weight"]
            # Update for multiclass
            params.update(
                {
                    "objective": "multi:softmax",
                    "num_class": num_classes,
                    "eval_metric": "merror",  # Metric for multiclass
                }
            )
        else:
            # Remove num_class if exists
            if "num_class" in params:
                del params["num_class"]
            # Update for binary classification
            params.update(
                {
                    "objective": "binary:logistic",
                    "scale_pos_weight": 3,  # Adjustment for class imbalance
                    "eval_metric": "error",  # Metric for binary classification
                }
            )

        # Save updated parameters to the settings file
        config.set_value("model_parameters", params)

        # Convert preview to list of dictionaries
        preview = df.head(10).to_dict(orient="records")
        
        return f"File {safe_filename} loaded successfully", preview
    except Exception as e:
        logger.error(f"Error loading file {safe_filename}: {e}")
        raise


async def upload_file(filename: str, contents: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Upload a new data file.
    
    Args:
        filename (str): Name of the file to upload
        contents (bytes): File contents
    
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (message, preview data)
    """
    # Sanitize filename to prevent path traversal
    safe_filename = sanitize_filename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    # Ensure DATA_DIR exists
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        # Write the file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)
            
        # After upload, load the file and update the configuration
        df = pd.read_csv(file_path)
        preview = df.head(10).to_dict(orient="records")

        # Update the config with the new file settings
        config.set_value("selected_file", safe_filename)
        config.set_value("loaded_data_path", file_path)
        
        logger.info(f"File uploaded successfully: {safe_filename}")
        
        return f"Successfully uploaded and loaded {safe_filename}", preview
    except Exception as e:
        logger.error(f"Error uploading file {safe_filename}: {e}")
        if os.path.exists(file_path):
            # Clean up if there was an error
            try:
                os.remove(file_path)
            except:
                pass
        raise


def delete_file(filename: str) -> str:
    """
    Delete a data file.
    
    Args:
        filename (str): Name of the file to delete
    
    Returns:
        str: Success message
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    # Sanitize filename to prevent path traversal
    safe_filename = sanitize_filename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    if not os.path.isfile(file_path):
        logger.error(f"File not found: {safe_filename}")
        raise FileNotFoundError(file_path)

    try:
        os.remove(file_path)
        
        # Update the configuration if the deleted file was selected/loaded
        current_selected = config.get_value("selected_file")
        if current_selected == safe_filename:
            config.set_value("selected_file", "")
            config.set_value("loaded_data_path", "")
            
        logger.info(f"File deleted successfully: {safe_filename}")
        
        return f"File {safe_filename} has been deleted."
    except Exception as e:
        logger.error(f"Error deleting file {safe_filename}: {e}")
        raise 