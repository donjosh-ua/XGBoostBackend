"""
Utility functions that can be used throughout the application.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.logging import app_logger as logger


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to ensure it's safe for file operations.
    
    Args:
        filename (str): The filename to sanitize
        
    Returns:
        str: Sanitized filename
    """
    return os.path.basename(filename)


def plot_accuracy_lines_and_curves(normal_results: Dict, custom_results: Dict, 
                                  output_path: str) -> str:
    """
    Plot accuracy comparison between normal and custom models.
    
    Args:
        normal_results (Dict): Results from normal model training
        custom_results (Dict): Results from custom model training
        output_path (str): Path to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(output_path))
    
    try:
        plt.figure(figsize=(12, 6))
        
        # Extract data from results
        normal_train = normal_results.get("train", {}).get("error", [])
        normal_test = normal_results.get("test", {}).get("error", [])
        
        custom_train = custom_results.get("train", {}).get("error", [])
        custom_test = custom_results.get("test", {}).get("error", [])
        
        # Plot data
        plt.subplot(1, 2, 1)
        plt.plot(normal_train, label="Normal - Train")
        plt.plot(normal_test, label="Normal - Test")
        plt.plot(custom_train, label="Custom - Train")
        plt.plot(custom_test, label="Custom - Test")
        plt.xlabel("Rounds")
        plt.ylabel("Error")
        plt.title("Error Comparison")
        plt.legend()
        plt.grid(True)
        
        # Distribution/histogram plot
        plt.subplot(1, 2, 2)
        
        # Convert to numpy arrays if they're not already
        if normal_test:
            normal_errors = np.array(normal_test)
            plt.hist(normal_errors, alpha=0.5, label="Normal Model", bins=10)
            
        if custom_test:
            custom_errors = np.array(custom_test)
            plt.hist(custom_errors, alpha=0.5, label="Custom Model", bins=10)
            
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.legend()
        plt.grid(True)
        
        # Save and close
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Accuracy comparison plot saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating accuracy plot: {e}")
        return ""


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         output_path: str, labels: Optional[List[str]] = None) -> str:
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        output_path (str): Path to save the confusion matrix plot
        labels (Optional[List[str]], optional): Class labels. Defaults to None.
        
    Returns:
        str: Path to the saved confusion matrix plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(output_path))
    
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")
        return ""


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    try:
        # Convert probabilities to class labels if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        elif y_true.dtype == np.int64 or y_true.dtype == np.int32:
            y_pred_labels = (y_pred >= 0.5).astype(int)
        else:
            y_pred_labels = y_pred
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_labels)),
            "precision": float(precision_score(y_true, y_pred_labels, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_labels, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_labels, average="weighted", zero_division=0))
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        } 