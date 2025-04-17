"""
XGBoost model service.
Provides core XGBoost functionality for training, prediction, and evaluation.
"""
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from app.core import config
from app.common.data_handling import load_data_from_csv, save_model
from app.common.utils import calculate_metrics
from app.core.logging import xgboost_logger as logger


def train_normal_xgboost(
    data_path: str, 
    params: Dict[str, Any], 
    method: str = "split", 
    num_folds: int = 5
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """
    Train a normal XGBoost model using split or cross validation.
    
    Args:
        data_path (str): Path to the training data
        params (Dict[str, Any]): XGBoost parameters
        method (str, optional): Training method ('split' or 'cv'). Defaults to "split".
        num_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
    
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: (trained model, evaluation results)
        
    Raises:
        ValueError: If the training method is invalid
    """
    rounds = config.get_value("rounds", 5)
    folds = config.get_value("training_value", 5) if method == "cv" else num_folds
    
    # Ensure output directory exists
    output_dir = os.path.join("app", "data", "models", "xgboost")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model_normal.xgb")
    
    if method == "split":
        # Train with train/test split
        dtrain, dtest, _, _, _, _ = load_data_from_csv()
        evals_result = {}
        
        logger.info("Training XGBoost model with train/test split")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
        )
        
        # Save the model
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, evals_result
    
    elif method == "cv":
        # Train with cross-validation
        df = pd.read_csv(data_path)
        if df.shape[1] < 2:
            error_msg = "Dataset must have at least one feature column and one target column"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Set up k-fold cross-validation
        seed = config.get_value("kseed", 1994)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        evals_result = {}
        booster = None
        
        logger.info(f"Training XGBoost model with {folds}-fold cross-validation")
        
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold+1}/{folds}")
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            dtrain_fold = xgb.DMatrix(X_train, label=y_train)
            dtest_fold = xgb.DMatrix(X_test, label=y_test)
            
            # Train the model for this fold
            fold_evals_result = {}
            booster = xgb.train(
                params,
                dtrain_fold,
                num_boost_round=rounds,
                evals=[(dtrain_fold, "train"), (dtest_fold, "test")],
                evals_result=fold_evals_result,
            )
            
            # Store the results
            for dataset in fold_evals_result:
                for metric in fold_evals_result[dataset]:
                    key = f"{dataset}_{metric}_fold{fold+1}"
                    evals_result[key] = fold_evals_result[dataset][metric]
        
        # Save the final model
        booster.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return booster, evals_result
    
    else:
        error_msg = f"Invalid training method: {method}. Use 'split' or 'cv'."
        logger.error(error_msg)
        raise ValueError(error_msg)


def predict_with_model(
    model_path: str, 
    data: Union[str, np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Make predictions with a trained XGBoost model.
    
    Args:
        model_path (str): Path to the trained model
        data (Union[str, np.ndarray, pd.DataFrame]): Data for prediction
            - If str: Path to a CSV file
            - If np.ndarray or pd.DataFrame: Feature data
    
    Returns:
        np.ndarray: Predictions
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the data format is invalid
    """
    # Check if model exists
    if not os.path.exists(model_path):
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load the model
    model = xgb.Booster(model_file=model_path)
    
    # Process input data
    if isinstance(data, str):
        # Data is a file path
        if not os.path.exists(data):
            error_msg = f"Data file not found: {data}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Load data from CSV
        try:
            df = pd.read_csv(data)
            dmatrix = xgb.DMatrix(df.values)
        except Exception as e:
            error_msg = f"Error loading data from {data}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    elif isinstance(data, np.ndarray):
        # Data is a numpy array
        dmatrix = xgb.DMatrix(data)
    
    elif isinstance(data, pd.DataFrame):
        # Data is a pandas DataFrame
        dmatrix = xgb.DMatrix(data.values)
    
    else:
        error_msg = f"Unsupported data type: {type(data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Make predictions
    logger.info("Making predictions with XGBoost model")
    predictions = model.predict(dmatrix)
    
    return predictions


def evaluate_model(
    model_path: str, 
    data_path: str
) -> Dict[str, float]:
    """
    Evaluate a trained XGBoost model.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the evaluation data
    
    Returns:
        Dict[str, float]: Evaluation metrics
        
    Raises:
        FileNotFoundError: If the model or data file doesn't exist
    """
    # Check if files exist
    if not os.path.exists(model_path):
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    if not os.path.exists(data_path):
        error_msg = f"Data file not found: {data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Load the model
        model = xgb.Booster(model_file=model_path)
        
        # Load data
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values
        
        # Make predictions
        dmatrix = xgb.DMatrix(X)
        y_pred_prob = model.predict(dmatrix)
        
        # Convert probabilities to class labels
        if len(y_pred_prob.shape) > 1:  # multiclass
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:  # binary
            y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        error_msg = f"Error evaluating model: {e}"
        logger.error(error_msg)
        raise


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    n_splits: int = 5,
    is_multiclass: bool = False
) -> Dict[str, float]:
    """
    Perform cross-validation with an XGBoost model.
    
    Args:
        X (np.ndarray): Feature data
        y (np.ndarray): Target data
        params (Dict[str, Any]): XGBoost parameters
        n_splits (int, optional): Number of CV folds. Defaults to 5.
        is_multiclass (bool, optional): Whether it's a multiclass problem. Defaults to False.
    
    Returns:
        Dict[str, float]: Cross-validation metrics
    """
    # Set up k-fold cross-validation
    seed = config.get_value("kseed", 1994)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Metrics for each fold
    accuracies, precisions, recalls, f1s = [], [], [], []
    
    logger.info(f"Performing {n_splits}-fold cross-validation")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        logger.info(f"Training fold {fold+1}/{n_splits}")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create XGBoost matrices
        dtrain_fold = xgb.DMatrix(X_train, label=y_train)
        dtest_fold = xgb.DMatrix(X_test, label=y_test)
        
        # Train the model for this fold
        rounds = config.get_value("rounds", 50)
        evals_result = {}
        booster = xgb.train(
            params,
            dtrain_fold,
            num_boost_round=rounds,
            evals=[(dtest_fold, "test")],
            evals_result=evals_result,
        )
        
        # Make predictions
        preds = booster.predict(dtest_fold)
        
        # Convert probabilities to class labels
        if is_multiclass:
            pred_labels = np.argmax(preds, axis=1)
        else:
            pred_labels = (preds >= 0.5).astype(int)
        
        # Calculate metrics
        fold_metrics = calculate_metrics(y_test, pred_labels)
        
        # Store metrics
        accuracies.append(fold_metrics["accuracy"])
        precisions.append(fold_metrics["precision"])
        recalls.append(fold_metrics["recall"])
        f1s.append(fold_metrics["f1"])
    
    # Aggregate metrics across folds
    cv_metrics = {
        "accuracy": float(np.mean(accuracies)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_std": float(np.std(precisions)),
        "recall_std": float(np.std(recalls)),
        "f1_std": float(np.std(f1s))
    }
    
    logger.info(f"Cross-validation metrics: {cv_metrics}")
    return cv_metrics


def grid_search_xgboost() -> Dict[str, Any]:
    """
    Perform grid search to find optimal XGBoost parameters.
    
    Returns:
        Dict[str, Any]: Best parameters found
    """
    # Load data
    _, _, train_x, train_y, _, _ = load_data_from_csv()
    
    # Define parameter grid (reduced for speed)
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "gamma": [0.01, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [1, 3],
        "scale_pos_weight": [1, 3]
    }
    
    logger.info("Performing grid search for XGBoost parameters")
    
    # Create XGBoost classifier
    seed = config.get_value("kseed", 1994)
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=seed)
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=1, 
        scoring="accuracy"
    )
    
    # Fit the grid search
    grid_search.fit(train_x, train_y)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logger.info(f"Best parameters found: {best_params}")
    
    return best_params 