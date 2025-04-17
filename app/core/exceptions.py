"""
Custom exceptions for the application.
Provides specific exception types for different error scenarios.
"""
from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Base exception for all application-specific exceptions."""
    
    def __init__(self, message: str = "An application error occurred"):
        self.message = message
        super().__init__(self.message)


class ConfigurationError(BaseAppException):
    """Raised when there's an issue with the configuration."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message)


class DataError(BaseAppException):
    """Base class for data-related errors."""
    
    def __init__(self, message: str = "Data error"):
        super().__init__(message)


class FileNotFoundError(DataError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: str):
        super().__init__(f"File not found: {file_path}")
        self.file_path = file_path


class InvalidDataFormatError(DataError):
    """Raised when data is in an invalid format."""
    
    def __init__(self, message: str = "Invalid data format"):
        super().__init__(message)


class ModelError(BaseAppException):
    """Base class for model-related errors."""
    
    def __init__(self, message: str = "Model error"):
        super().__init__(message)


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""
    
    def __init__(self, model_path: str):
        super().__init__(f"Model not found: {model_path}")
        self.model_path = model_path


class TrainingError(ModelError):
    """Raised when there's an error during model training."""
    
    def __init__(self, message: str = "Error during model training"):
        super().__init__(message)


class PredictionError(ModelError):
    """Raised when there's an error during model prediction."""
    
    def __init__(self, message: str = "Error during model prediction"):
        super().__init__(message)


class ValidationError(BaseAppException):
    """Raised when validation fails."""
    
    def __init__(self, message: str = "Validation error", errors: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.errors = errors or {} 