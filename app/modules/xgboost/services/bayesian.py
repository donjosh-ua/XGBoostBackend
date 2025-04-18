"""
Bayesian optimization service for XGBoost.
Provides custom objective functions and PyMC-based adjustments.
"""
import os
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import pymc as pm
from scipy.special import expit
from sklearn.model_selection import KFold

from app.core import config
from app.common.data_handling import load_data_from_csv, save_model
from app.core.logging import xgboost_logger as logger


def get_distribution_params() -> Dict[str, Any]:
    """
    Get parameters for the selected distribution from configuration.
    
    Returns:
        Dict[str, Any]: Distribution parameters
    """
    custom_params = config.get_value("custom_parameters", {})
    if not custom_params:
        # Default parameters
        custom_params = {
            "mean": 0,
            "sigma": 10,
            "alpha": 0,
            "beta": 0,
            "lambda": 0
        }
    
    distribution = config.get_value("distribution", "Normal")
    
    distribution_params = {
        "Normal": {"mu": custom_params.get("mean", 0), "sigma": custom_params.get("sigma", 10)},
        "HalfNormal": {"sigma": custom_params.get("sigma", 10)},
        "Cauchy": {"alpha": custom_params.get("alpha", 0), "beta": custom_params.get("beta", 0)},
        "Exponential": {"lam": custom_params.get("lambda", 1)},
    }
    
    return distribution_params.get(distribution, {"mu": 0, "sigma": 1})


def apply_pymc_adjustment(preds: np.ndarray) -> np.ndarray:
    """
    Adjust predictions with PyMC using the selected distribution.
    Supports binary (1D) or multiclass (2D) by applying logistic or softmax.
    
    Args:
        preds (np.ndarray): Raw predictions
        
    Returns:
        np.ndarray: Adjusted predictions
    """
    preds = np.array(preds)
    params = get_distribution_params()
    distribution = config.get_value("distribution", "Normal")
    
    logger.info(f"Applying PyMC adjustment with {distribution} distribution")
    
    # Distribution factory functions
    distributions = {
        "Normal": lambda name, params, shape: pm.Normal(
            name, mu=params.get("mu", 0), sigma=params.get("sigma", 1), shape=shape
        ),
        "HalfNormal": lambda name, params, shape: pm.HalfNormal(
            name, sigma=params.get("sigma", 1), shape=shape
        ),
        "Cauchy": lambda name, params, shape: pm.Cauchy(
            name, alpha=params.get("alpha", 0), beta=params.get("beta", 1), shape=shape
        ),
        "Exponential": lambda name, params, shape: pm.Exponential(
            name, lam=params.get("lam", 1), shape=shape
        ),
    }
    
    # Detect if binary or multiclass
    is_multiclass = preds.ndim > 1
    shape = preds.shape[1] if is_multiclass else None
    
    # Get the distribution function
    dist_func = distributions.get(distribution)
    if dist_func is None:
        error_msg = f"Distribution '{distribution}' not supported."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        with pm.Model() as model:
            # Create adjustment distribution
            adjustment = dist_func("adjustment", params, shape=shape)
            adjusted_logits = preds + adjustment
            
            # Sample from the posterior
            trace = pm.sample(
                1000, 
                tune=1000, 
                chains=2, 
                cores=2, 
                progressbar=False,
                return_inferencedata=True
            )
            
            # Process the results based on problem type
            if is_multiclass:
                # Get the mean adjustment
                adjustment_samples = trace.posterior["adjustment"].mean(dim=("chain", "draw")).values
                
                # Apply adjustment and softmax
                adjusted_logits = preds + adjustment_samples
                exp_vals = np.exp(adjusted_logits)
                adjusted_preds = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
            else:
                # Get the mean adjustment
                adjustment_samples = trace.posterior["adjustment"].mean().values
                
                # Apply adjustment and sigmoid
                adjusted_logits = preds + adjustment_samples
                adjusted_preds = expit(adjusted_logits)
            
            return adjusted_preds
    except Exception as e:
        error_msg = f"Error in PyMC adjustment: {e}"
        logger.error(error_msg)
        logger.info("Falling back to original predictions")
        return preds


def custom_objective_factory() -> Callable:
    """
    Create a custom objective function for XGBoost with the selected distribution.
    
    Returns:
        Callable: Custom objective function
    """
    def custom_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function that applies PyMC adjustment.
        
        Args:
            preds (np.ndarray): Predictions
            dtrain (xgb.DMatrix): Training data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradient and Hessian
        """
        try:
            # Apply PyMC adjustment
            adjusted_preds = apply_pymc_adjustment(preds)
            
            # Get labels
            labels = dtrain.get_label()
            
            # Calculate gradient and hessian
            if preds.ndim == 1:
                # Binary classification
                grad = adjusted_preds - labels
                hess = adjusted_preds * (1 - adjusted_preds)
            else:
                # Multiclass classification (approximation)
                rows = len(labels)
                grad = np.copy(adjusted_preds)
                grad[np.arange(rows), labels.astype(int)] -= 1.0
                hess = adjusted_preds * (1.0 - adjusted_preds)
            
            return grad, hess
        except Exception as e:
            error_msg = f"Error in custom objective function: {e}"
            logger.error(error_msg)
            
            # Fallback to logistic regression gradient/hessian
            if preds.ndim == 1:
                # Binary logistic regression
                preds_as_prob = expit(preds)
                grad = preds_as_prob - labels
                hess = preds_as_prob * (1 - preds_as_prob)
            else:
                # Multiclass softmax
                rows = len(labels)
                exp_preds = np.exp(preds)
                softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
                grad = np.copy(softmax_preds)
                grad[np.arange(rows), labels.astype(int)] -= 1.0
                hess = softmax_preds * (1.0 - softmax_preds)
            
            return grad, hess
    
    return custom_objective


def train_custom_xgboost(
    data_path: str,
    params: Dict[str, Any],
    distribution: str,
    method: str = "split",
    num_folds: int = 5,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """
    Train an XGBoost model with a custom objective function.
    
    Args:
        data_path (str): Path to the training data
        params (Dict[str, Any]): XGBoost parameters
        distribution (str): Distribution for the custom objective
        method (str, optional): Training method ('split' or 'cv'). Defaults to "split".
        num_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: (trained model, evaluation results)
        
    Raises:
        ValueError: If the training method is invalid
    """
    # Set the distribution in config
    config.set_value("distribution", distribution)
    
    rounds = config.get_value("rounds", 5)
    folds = config.get_value("training_value", 5) if method == "cv" else num_folds
    
    # Create custom objective function
    custom_obj = custom_objective_factory()
    
    # Ensure output directory exists
    output_dir = os.path.join("app", "data", "models", "xgboost")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model_custom.xgb")
    
    logger.info(f"Training XGBoost model with custom objective ({distribution} distribution)")
    
    if method == "split":
        # Train with train/test split
        dtrain, dtest, _, _, _, _ = load_data_from_csv()
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            obj=custom_obj,
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
                obj=custom_obj,
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