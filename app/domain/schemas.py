"""
Pydantic schemas for API requests and responses.
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


# Common schemas
class MessageResponse(BaseModel):
    """Standard response with a message."""
    message: str


# Data management schemas
class FileSelection(BaseModel):
    """Schema for selecting a file."""
    filename: str


class DataLoadRequest(BaseModel):
    """Schema for loading a data file."""
    filename: str
    has_header: bool = Field(default=False, description="Whether the file has a header row")
    separator: str = Field(default=",", description="Column separator character")


class DataPreviewResponse(BaseModel):
    """Response with a data preview."""
    message: str
    preview: List[Dict[str, Any]]


class AvailableFilesResponse(BaseModel):
    """Response with available files."""
    files: List[str]


# XGBoost schemas
class XGBoostParameters(BaseModel):
    """XGBoost model parameters."""
    eta: Optional[float] = Field(default=0.05, description="Learning rate")
    max_depth: Optional[int] = Field(default=5, description="Maximum tree depth")
    gamma: Optional[float] = Field(default=0.1, description="Minimum loss reduction for partition")
    learning_rate: Optional[float] = Field(default=0.1, description="Learning rate")
    min_child_weight: Optional[int] = Field(default=3, description="Minimum sum of instance weight in a child")
    subsample: Optional[float] = Field(default=0.7, description="Subsample ratio of training data")
    colsample_bytree: Optional[float] = Field(default=1.0, description="Subsample ratio of columns")
    scale_pos_weight: Optional[float] = Field(default=None, description="Weight of positive class for imbalanced datasets")
    objective: Optional[str] = Field(default="binary:logistic", description="Objective function")
    eval_metric: Optional[str] = Field(default="error", description="Evaluation metric")
    seed: Optional[int] = Field(default=1994, description="Random seed")


class TrainRequest(BaseModel):
    """Request to train a model."""
    method: str = Field(description="Training method: 'split' or 'cv'")
    value: int = Field(description="If method='cv': number of folds; if method='split': percentage of test set")
    rounds: int = Field(default=5, description="Number of training rounds")
    distribution: Optional[str] = Field(default=None, description="Distribution for custom training")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Additional model parameters")


class TrainResponse(BaseModel):
    """Response to a training request."""
    message: str
    metrics: Optional[Dict[str, float]] = None


# Prediction schemas
class PredictRequest(BaseModel):
    """Request to make a prediction."""
    model_path: str
    data_path: str


class PredictResponse(BaseModel):
    """Response with predictions."""
    predictions: List[float]


class PredictWithDataRequest(BaseModel):
    """Request to make a prediction with inline data."""
    model_path: str
    data: List[List[float]]


# Evaluation schemas
class EvaluateRequest(BaseModel):
    """Request to evaluate a model."""
    model_path: str
    data_path: str


class EvaluateResponse(BaseModel):
    """Response with evaluation metrics."""
    metrics: Dict[str, float]


# Hyperparameter tuning schemas
class TuningRequest(BaseModel):
    """Request for hyperparameter tuning."""
    method: str = Field(description="Tuning method: 'grid', 'bayesian'")
    param_grid: Optional[Dict[str, List[Any]]] = None
    search_space: Optional[Dict[str, Dict[str, Any]]] = None
    cv_folds: int = Field(default=3, description="Number of cross-validation folds")


class TuningResponse(BaseModel):
    """Response with tuning results."""
    message: str
    best_params: Dict[str, Any]
    best_score: float


# Neural Network schemas
class NeuralNetworkConfig(BaseModel):
    """Neural network configuration."""
    alpha: float = Field(default=0.001, description="Learning rate")
    epoch: int = Field(default=100, description="Number of epochs")
    criteria: str = Field(default="cross_entropy", description="Loss function")
    optimizer: str = Field(default="SGD", description="Optimizer algorithm")
    hidden_layers: List[int] = Field(default=[], description="Sizes of hidden layers")
    activation: str = Field(default="Tanh", description="Activation function")
    momentum: float = Field(default=0.9, description="Momentum for SGD")
    decay: float = Field(default=0.0, description="Learning rate decay")
    batch_size: int = Field(default=64, description="Batch size")
    image: bool = Field(default=False, description="Whether data is image data")
    image_size: Optional[int] = Field(default=None, description="Image size for flattening")
    Bay: bool = Field(default=False, description="Whether to use Bayesian approach")
    Lambda: float = Field(default=0.005, description="Regularization parameter")


class TrainNNRequest(BaseModel):
    """Request to train a neural network."""
    method: str = Field(description="Training method: 'split' or 'cv'")
    value: int = Field(description="If method='cv': number of folds; if method='split': percentage of test split")
    config: NeuralNetworkConfig


class PredictNNRequest(BaseModel):
    """Request to make a prediction with a neural network."""
    data: List[List[float]] = Field(description="Input data for prediction")
    image: bool = Field(default=False, description="Whether data is image")
    image_size: Optional[int] = Field(default=None, description="Size of flattened image if applicable") 