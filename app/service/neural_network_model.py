import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import the RedNeuBay and related modules
import sys

# Add the parent directory to sys.path to find the XGBoostRNA module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.XGBoostRNA.redneuronal_bay.RedNeuBay import RedNeuBay
from app.XGBoostRNA.redneuronal_bay.Layers.layers import *
from app.XGBoostRNA.redneuronal_bay.Div_Datos import (
    trat_Dat,
    cv_prepros,
    cv_trat_Dat,
    trat_Imag,
)
from app.XGBoostRNA.redneuronal_bay.utils import *

# Define constants
OUTPUT_FOLDER = "app/data/neural_network_models/"
PLOTS_FOLDER = "app/data/plots/"


def ensure_folders_exist():
    """Ensure that the necessary folders exist for saving models and plots."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)


def load_data_from_csv(data_path, target_column="class"):
    """Load data from CSV file and prepare for neural network training."""
    # Read the CSV file
    df = pd.read_csv(data_path)

    # Extract the target column
    if target_column not in df.columns:
        # Try to infer the target column as the last column
        target_column = df.columns[-1]

    # Prepare features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Basic information
    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    return df, X, y, num_features, num_classes


def create_network(config, input_dim, num_classes):
    """Create and configure a neural network based on the provided configuration."""
    # Create the RedNeuBay instance with the provided configuration
    network = RedNeuBay(
        alpha=config.get("alpha", 0.001),
        epoch=config.get("epoch", 100),
        criteria=config.get("criteria", "cross_entropy"),
        optimizer=config.get("optimizer", "SGD"),
        image_size=config.get("image_size", None),
        verbose=config.get("verbose", True),
        Lambda=config.get("Lambda", 0.005),
        decay=config.get("decay", 0.0),
        momentum=config.get("momentum", 0.9),
        image=config.get("image", False),
        FA_ext=config.get("FA_ext", None),
        Bay=config.get("Bay", False),
        save_mod=config.get("save_mod", "NNModel"),
        pred_hot=config.get("pred_hot", True),
        test_size=config.get("test_size", 0.2),
        batch_size=config.get("batch_size", 64),
        cv=config.get("cv", False),
        Kfold=config.get("Kfold", 5),
    )

    # Add layers based on configuration
    hidden_layers = config.get("hidden_layers", [])
    activation = config.get("activation", "Tanh")

    # If no hidden layers are specified, create a simple network with one hidden layer
    if not hidden_layers:
        hidden_dim = max(int(input_dim * 1.5), 10)
        hidden_layers = [hidden_dim]

    # Input layer
    if activation == "Tanh":
        network.add(Tanh_Layer(input_dim, hidden_layers[0]))
    elif activation == "Sigmoid":
        network.add(Sigmoid_Layer(input_dim, hidden_layers[0]))
    elif activation == "ReLU":
        network.add(ReLU_Layer(input_dim, hidden_layers[0]))

    # Hidden layers
    for i in range(len(hidden_layers) - 1):
        if activation == "Tanh":
            network.add(Tanh_Layer(hidden_layers[i], hidden_layers[i + 1]))
        elif activation == "Sigmoid":
            network.add(Sigmoid_Layer(hidden_layers[i], hidden_layers[i + 1]))
        elif activation == "ReLU":
            network.add(ReLU_Layer(hidden_layers[i], hidden_layers[i + 1]))

    # Output layer - always use Softmax for classification
    network.add(Softmax_Layer(hidden_layers[-1], num_classes))

    return network


def train_neural_network(data_path, config, method="split"):
    """Train a neural network using the provided configuration."""
    ensure_folders_exist()

    # Load the data
    df, X, y, num_features, num_classes = load_data_from_csv(data_path)

    # Create the network
    network = create_network(config, num_features, num_classes)

    # Modify the save_mod to include the current directory
    original_save_mod = network.save_mod
    network.save_mod = f"{OUTPUT_FOLDER}{original_save_mod}"

    # Train the network
    if method.lower() == "cv":
        # Cross-validation training
        network.cv = True
        result = network.cv_train(X_cla=X, Y_cla=y)

        # Calculate accuracy across all folds
        accuracies = []
        for k in range(network.Kfold):
            model_path = f"{OUTPUT_FOLDER}best_{original_save_mod}_K{k+1}"
            if os.path.exists(model_path):
                # Load the model
                model = torch.load(model_path)
                # Use a portion of the data for testing
                test_indices = np.random.choice(
                    len(X), size=int(len(X) * 0.2), replace=False
                )
                X_test, y_test = X[test_indices], y[test_indices]
                # Predict
                accuracy = predict_with_neural_network(
                    model, X_test, y_test, image=False, image_size=None
                )
                accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies) if accuracies else 0
        return result, {"accuracy": mean_accuracy}
    else:
        # Split training
        network.cv = False
        result = network.train(X_cla=X, Y_cla=y)

        # Load the trained model for evaluation
        model_path = f"{OUTPUT_FOLDER}best_{original_save_mod}"
        if os.path.exists(model_path):
            model = torch.load(model_path)
            # Use a portion of the data for testing
            test_indices = np.random.choice(
                len(X), size=int(len(X) * 0.2), replace=False
            )
            X_test, y_test = X[test_indices], y[test_indices]
            # Predict
            accuracy = predict_with_neural_network(
                model, X_test, y_test, image=False, image_size=None
            )
            return result, {"accuracy": accuracy}

        return result, {"accuracy": 0}


def predict_with_neural_network(model, X, y=None, image=False, image_size=None):
    """Make predictions using a trained neural network model."""
    outputs = []

    if image:
        if isinstance(X, torch.Tensor):
            enput = X.view(-1, image_size)
        else:
            enput = torch.tensor(X).view(-1, image_size)
    else:
        enput = torch.FloatTensor(X)
        if y is not None:
            y = torch.Tensor(y)

    # Forward pass through the model
    for i in range(len(model)):
        layer = model[i]
        a = 1
        Output = layer.funcion_activacion(
            torch.add(torch.matmul(enput, layer.weights), layer.bias), a
        )
        Output = torch.FloatTensor(Output)
        outputs = Output
        enput = Output

    # Get predicted class
    _, pred = torch.max(outputs, 1)

    if y is None:
        return pred.numpy()
    else:
        # Calculate accuracy
        n_total_row = len(y)
        accuracy = torch.sum(pred == y).float() / n_total_row
        return accuracy.item()


def save_confusion_matrix(y_true, y_pred, save_path):
    """Create and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    # Get unique classes
    classes = np.unique(y_true)

    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap=plt.cm.Blues, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def compare_neural_networks(normal_results, custom_results, save_path):
    """Create and save comparison plots for different neural network configurations."""
    plt.figure(figsize=(12, 6))

    # Plot accuracies
    plt.subplot(1, 2, 1)
    labels = ["Normal NN", "Custom NN"]
    accuracies = [normal_results.get("accuracy", 0), custom_results.get("accuracy", 0)]
    plt.bar(labels, accuracies, color=["blue", "orange"])
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.05, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path
