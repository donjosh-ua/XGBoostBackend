"""
Neural network model service.
Handles training and prediction with neural network models.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.logging import nn_logger as logger
from app.core.exceptions import ModelError, TrainingError


class SimpleNeuralNetwork(nn.Module):
    """A simple neural network model."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, activation: str = "ReLU"):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): The number of input features
            hidden_size (int, optional): The number of hidden neurons. Defaults to 64.
            output_size (int, optional): The number of output neurons. Defaults to 1.
            activation (str, optional): Activation function to use. Defaults to "ReLU".
        """
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        
        # Set activation function based on config
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
            
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        if self.layer2.out_features == 1:  # Binary classification
            x = self.sigmoid(x)
        return x


class BayesianNeuralNetwork(PyroModule):
    """A Bayesian neural network model."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, activation: str = "ReLU"):
        """
        Initialize the Bayesian neural network.
        
        Args:
            input_size (int): The number of input features
            hidden_size (int, optional): The number of hidden neurons. Defaults to 64.
            output_size (int, optional): The number of output neurons. Defaults to 1.
            activation (str, optional): Activation function to use. Defaults to "ReLU".
        """
        super(BayesianNeuralNetwork, self).__init__()
        
        # Define priors for weights and biases
        self.layer1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.layer1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
        
        # Set activation function based on config
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
            
        self.layer2 = PyroModule[nn.Linear](hidden_size, output_size)
        self.layer2.weight = PyroSample(dist.Normal(0., 1.).expand([output_size, hidden_size]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., 1.).expand([output_size]).to_event(1))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            y (Optional[torch.Tensor], optional): Target tensor for Bayesian inference. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        
        if self.layer2.out_features == 1:  # Binary classification
            x = self.sigmoid(x)
            
        # Handle Bayesian inference
        if y is not None:
            # Flatten outputs and targets for binary classification
            x_flat = x.view(-1)
            y_flat = y.view(-1)
            
            # Likelihood for binary classification
            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Bernoulli(probs=x_flat), obs=y_flat)
        
        return x


def get_criterion(criteria: str) -> nn.Module:
    """
    Get the appropriate loss function based on the criteria.
    
    Args:
        criteria (str): Loss function name
        
    Returns:
        nn.Module: PyTorch loss function
    """
    if criteria == "cross_entropy":
        return nn.BCELoss()
    elif criteria == "mse":
        return nn.MSELoss()
    else:
        return nn.BCELoss()


def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, momentum: float = 0.9) -> optim.Optimizer:
    """
    Get the appropriate optimizer based on the name.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer_name (str): Optimizer name
        learning_rate (float): Learning rate
        momentum (float, optional): Momentum for SGD. Defaults to 0.9.
        
    Returns:
        optim.Optimizer: PyTorch optimizer
    """
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for the model.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels or probabilities
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Convert probabilities to class labels if needed
    if y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = (y_pred >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_labels)),
        "precision": float(precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)),
    }
    
    # Add ROC AUC if binary classification
    if len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except:
            metrics["roc_auc"] = 0.5
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="neural_network", is_bayesian=False):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
        model_name (str): Name of the model for the plot filename
        is_bayesian (bool): Whether this is a Bayesian neural network
        
    Returns:
        str: Path to the saved plot file
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join("app", "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert probabilities to class labels
    if y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = (y_pred >= 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot as a heatmap with seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{'Bayesian ' if is_bayesian else ''}Neural Network Confusion Matrix")
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Confusion matrix plot saved to {plot_path}")
    return plot_path


def plot_training_history(losses, model_name="neural_network", is_bayesian=False, accuracy=None):
    """
    Create and save training history plots.
    
    Args:
        losses (list): List of training losses
        model_name (str): Name of the model for the plot filename
        is_bayesian (bool): Whether this is a Bayesian neural network
        accuracy (list, optional): List of training accuracies
        
    Returns:
        str: Path to the saved plot file
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join("app", "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(losses, label=f"{'Bayesian ' if is_bayesian else ''}Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{'Bayesian ' if is_bayesian else ''}Neural Network Training Loss")
    plt.grid(True)
    plt.legend()
    
    # Accuracy plot if available
    plt.subplot(1, 2, 2)
    if accuracy is not None:
        plt.plot(accuracy, label=f"{'Bayesian ' if is_bayesian else ''}Accuracy")
    else:
        # Just show loss trend in the second plot
        plt.plot(losses, label=f"{'Bayesian ' if is_bayesian else ''}Loss Trend")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(f"{'Bayesian ' if is_bayesian else ''}Neural Network {'Accuracy' if accuracy else 'Loss Trend'}")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{model_name}_history.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Training history plot saved to {plot_path}")
    return plot_path


def train_neural_network(data_path: str, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a neural network model on the given data.
    
    Args:
        data_path (str): Path to the data file
        config (Dict[str, Any]): Neural network configuration
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Trained model and training metrics
        
    Raises:
        TrainingError: If there's an error during training
    """
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        
        # Get configuration options
        has_header = config.get("has_header", False)
        separator = config.get("separator", ",")
        header = 0 if has_header else None
        
        # Load data
        df = pd.read_csv(data_path, header=header, sep=separator)
        
        # Get features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Split data if needed
        test_size = config.get("test_size", 0.2)
        seed = config.get("seed", 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Determine model dimensions
        input_size = X.shape[1]
        output_size = 1  # Binary classification for now
        hidden_size = config.get("hidden_size", 64)
        activation = config.get("activation", "ReLU")
        
        # Check if Bayesian approach is enabled
        use_bayesian = config.get("bayesian", False)
        
        if use_bayesian:
            logger.info("Using Bayesian Neural Network")
            # Initialize Bayesian neural network
            model = BayesianNeuralNetwork(input_size, hidden_size, output_size, activation)
            
            # Setup Pyro for Bayesian inference
            pyro.clear_param_store()
            guide = AutoDiagonalNormal(model)
            
            # Get learning rate and regularization parameter
            learning_rate = config.get("learning_rate", 0.01)
            Lambda = config.get("Lambda", 0.005)  # Regularization parameter
            
            # Setup SVI (Stochastic Variational Inference)
            optimizer = pyro.optim.Adam({"lr": learning_rate, "weight_decay": Lambda})
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
            
            # Number of epochs
            num_epochs = config.get("epochs", 100)
            batch_size = config.get("batch_size", 32)
            
            # Keep track of losses
            losses = []
            
            logger.info(f"Starting Bayesian neural network training for {num_epochs} epochs")
            
            # Training loop
            for epoch in range(num_epochs):
                # SVI step
                loss = svi.step(X_train_tensor, y_train_tensor)
                losses.append(loss)
                
                if (epoch+1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
            
            # Save the model
            model_dir = os.path.join("app", "data", "models", "neural_network")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "bayesian_nn.pt")
            torch.save((model, guide), model_path)
            
            # Make predictions for evaluation
            with torch.no_grad():
                try:
                    # Using the model directly for evaluation
                    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=100)
                    # Get the raw predictions 
                    pred_samples = predictive(X_test_tensor)
                    
                    # Log available keys for debugging
                    logger.info(f"Available keys in predictive result: {list(pred_samples.keys())}")
                    
                    # Try to find the best key to use for predictions
                    if "_RETURN" in pred_samples:
                        y_pred = pred_samples["_RETURN"].mean(0).numpy()
                    elif "obs" in pred_samples:
                        y_pred = pred_samples["obs"].mean(0).numpy()
                    else:
                        # Direct model inference as fallback
                        y_pred = model(X_test_tensor).detach().numpy()
                        logger.warning("Using direct model output as fallback for prediction")
                        
                except Exception as e:
                    logger.error(f"Error during Bayesian prediction: {e}")
                    # Use direct model evaluation as fallback
                    y_pred = model(X_test_tensor).detach().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            metrics["loss"] = losses
            
            # Get model name for visualization
            model_name = config.get("save_mod", "bayesian_nn")
            
            # Generate and save training history plot
            plot_training_history(losses, model_name, is_bayesian=True)
            
            # Generate and save confusion matrix plot
            plot_confusion_matrix(y_test, y_pred, model_name, is_bayesian=True)
            
            logger.info(f"Model saved to {model_path}")
            
            return (model, guide), metrics
            
        else:
            # Initialize standard neural network
            model = SimpleNeuralNetwork(input_size, hidden_size, output_size, activation)
            
            # Get loss function
            criteria = config.get("criteria", "cross_entropy")
            criterion = get_criterion(criteria)
            
            # Get optimizer
            learning_rate = config.get("learning_rate", 0.01)
            optimizer_name = config.get("optimizer", "SGD")
            momentum = config.get("momentum", 0.9)
            optimizer = get_optimizer(model, optimizer_name, learning_rate, momentum)
            
            # Train the model
            num_epochs = config.get("epochs", 100)
            batch_size = config.get("batch_size", 32)
            
            # Keep track of losses and accuracies
            losses = []
            accuracies = []
            
            logger.info(f"Starting neural network training for {num_epochs} epochs")
            
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record loss
                losses.append(loss.item())
                
                # Calculate accuracy periodically
                if epoch % 5 == 0:
                    with torch.no_grad():
                        predicted = (outputs >= 0.5).float()
                        correct = (predicted == y_train_tensor).sum().item()
                        accuracy = correct / y_train_tensor.size(0)
                        accuracies.append(accuracy)
                
                if (epoch+1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            # Save the model
            model_dir = os.path.join("app", "data", "models", "neural_network")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "simple_nn.pt")
            torch.save(model, model_path)
            
            # Make predictions for evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor).numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            metrics["loss"] = losses
            
            # Get model name for visualization
            model_name = config.get("save_mod", "simple_nn")
            
            # Generate and save training history plot
            if accuracies:
                plot_training_history(losses, model_name, is_bayesian=False, accuracy=accuracies)
            else:
                plot_training_history(losses, model_name, is_bayesian=False)
            
            # Generate and save confusion matrix plot
            plot_confusion_matrix(y_test, y_pred, model_name, is_bayesian=False)
            
            logger.info(f"Model saved to {model_path}")
            
            return model, metrics
    
    except Exception as e:
        logger.error(f"Error training neural network: {e}")
        raise TrainingError(f"Error training neural network: {e}")


def predict_with_model(model_path: str, data: Union[str, np.ndarray, pd.DataFrame], num_samples: int = 100) -> np.ndarray:
    """
    Make predictions with a trained neural network model.
    
    Args:
        model_path (str): Path to the saved model
        data (Union[str, np.ndarray, pd.DataFrame]): Input data
        num_samples (int, optional): Number of samples for Bayesian prediction. Defaults to 100.
        
    Returns:
        np.ndarray: Predictions
        
    Raises:
        ModelError: If there's an error during prediction
    """
    try:
        # Process input data
        if isinstance(data, str):
            # Load from file
            df = pd.read_csv(data)
            X = df.iloc[:, :-1].values
        elif isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Determine if it's a Bayesian model
        is_bayesian = False
        
        # Load model
        loaded = torch.load(model_path)
        if isinstance(loaded, tuple):
            # Bayesian model
            model, guide = loaded
            is_bayesian = True
        else:
            # Standard model
            model = loaded
            model.eval()
        
        # Make predictions
        if is_bayesian:
            try:
                pyro.clear_param_store()
                # Use predictive for Bayesian model
                predictive = pyro.infer.Predictive(model, guide=guide, num_samples=num_samples)
                # Get raw predictions
                pred_samples = predictive(X_tensor)
                
                # Log available keys for debugging
                logger.info(f"Available keys in predictive result: {list(pred_samples.keys())}")
                
                # Try different keys for predictions
                if "_RETURN" in pred_samples:
                    predictions = pred_samples["_RETURN"].mean(0).numpy()
                elif "obs" in pred_samples:
                    predictions = pred_samples["obs"].mean(0).numpy()
                else:
                    # Direct model inference as fallback
                    predictions = model(X_tensor).detach().numpy()
                    logger.warning("Using direct model output as fallback for prediction")
            except Exception as e:
                logger.error(f"Error during Bayesian prediction: {e}")
                # Use direct model evaluation as fallback
                predictions = model(X_tensor).detach().numpy()
        else:
            # Standard prediction
            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = outputs.numpy()
        
        # For binary classification, convert to class labels if needed
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
            
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise ModelError(f"Error making predictions: {e}")


def evaluate_model(model_path: str, data_path: str) -> Dict[str, float]:
    """
    Evaluate a neural network model on a dataset.
    
    Args:
        model_path (str): Path to the saved model
        data_path (str): Path to the evaluation data
        
    Returns:
        Dict[str, float]: Evaluation metrics
        
    Raises:
        ModelError: If there's an error during evaluation
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y_true = df.iloc[:, -1].values
        
        # Make predictions
        y_pred = predict_with_model(model_path, X)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise ModelError(f"Error evaluating model: {e}") 