import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.utils import conf_manager
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

plt.switch_backend("agg")


def get_number_of_classes():
    """Obtener el número de clases del archivo cargado."""
    loaded_path = conf_manager.get_value("loaded_data_path")

    header = conf_manager.get_value("has_header")
    if header is not None:
        header = 0  # first row header

    data = pd.read_csv(
        loaded_path, header=header, sep=conf_manager.get_value("separator")
    )

    return len(data.iloc[:, -1].unique())


def display_metrics(
    preds: np.ndarray,
    test_y: np.ndarray,
    is_multiclass: bool,
    title: str = "Métricas de Rendimiento",
    output_path: str = None,
):
    """Genera una figura con métricas de rendimiento y la guarda en output_path."""
    metrics = {
        "Accuracy": accuracy_score(test_y, preds),
        "Precision": precision_score(
            test_y, preds, average="weighted" if is_multiclass else "binary"
        ),
        "Recall": recall_score(
            test_y, preds, average="weighted" if is_multiclass else "binary"
        ),
        "F1 Score": f1_score(
            test_y, preds, average="weighted" if is_multiclass else "binary"
        ),
    }
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table = ax.table(
        cellText=[[metric, f"{value:.4f}"] for metric, value in metrics.items()],
        colLabels=["Métrica", "Valor"],
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.title(title, fontsize=16)
    if output_path:
        plt.savefig(output_path)
    plt.close(fig)


def plot_label_distributions_side_by_side(
    true_labels: np.ndarray,
    normal_preds: np.ndarray,
    adjusted_preds: np.ndarray,
    output_path: str = None,
):
    """Genera la comparación de distribuciones de etiquetas y la guarda en output_path."""
    unique_classes = np.unique(true_labels)
    true_counts = [np.sum(true_labels == cls) for cls in unique_classes]
    normal_counts = [np.sum(normal_preds == cls) for cls in unique_classes]
    adjusted_counts = [np.sum(adjusted_preds == cls) for cls in unique_classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_positions = np.arange(len(unique_classes))
    width = 0.3

    axes[0].bar(
        x_positions - width / 2, true_counts, width=width, label="True", color="blue"
    )
    axes[0].bar(
        x_positions + width / 2, normal_counts, width=width, label="Normal", color="red"
    )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(unique_classes)
    axes[0].set_title("Normal XGBoost Distribution")
    axes[0].legend()

    axes[1].bar(
        x_positions - width / 2, true_counts, width=width, label="True", color="blue"
    )
    axes[1].bar(
        x_positions + width / 2,
        adjusted_counts,
        width=width,
        label="Adjusted",
        color="red",
    )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(unique_classes)
    axes[1].set_title("PyMC-Adjusted Distribution")
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close(fig)


def show_confusion_matrices_side_by_side(
    true_labels: np.ndarray,
    normal_preds: np.ndarray,
    adjusted_preds: np.ndarray,
    output_path: str = None,
):
    """Genera dos matrices de confusión lado a lado y las guarda en output_path."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_normal = confusion_matrix(true_labels, normal_preds)
    disp_normal = ConfusionMatrixDisplay(confusion_matrix=cm_normal)
    disp_normal.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title("Normal XGBoost")

    cm_adjusted = confusion_matrix(true_labels, adjusted_preds)
    disp_adjusted = ConfusionMatrixDisplay(confusion_matrix=cm_adjusted)
    disp_adjusted.plot(ax=axes[1], cmap=plt.cm.Blues)
    axes[1].set_title("PyMC-Adjusted")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close(fig)


def plot_accuracy_lines_and_curves(
    normal_evals_result, custom_evals_result, output_path: str = None
):
    """
    Muestra dos curvas de precisión de entrenamiento (Normal y Bayesiano)
    y dos líneas horizontales con la precisión final de test (Normal y Bayesiano),
    y guarda el gráfico en output_path si se proporciona.
    """
    is_multiclass = get_number_of_classes() > 2
    metric = "merror" if is_multiclass else "error"

    # Entrenamiento
    normal_train_err = normal_evals_result["train"][metric]
    custom_train_err = custom_evals_result["train"][metric]
    normal_train_acc = [1.0 - e for e in normal_train_err]
    custom_train_acc = [1.0 - e for e in custom_train_err]

    # Test
    normal_test_err = normal_evals_result["test"][metric]
    custom_test_err = custom_evals_result["test"][metric]
    normal_final_test_acc = 1.0 - normal_test_err[-1]
    custom_final_test_acc = 1.0 - custom_test_err[-1]

    rounds = range(len(normal_train_acc))

    plt.figure(figsize=(8, 6))

    # Curvas de training accuracy
    plt.plot(rounds, normal_train_acc, label="Normal Accuracy", color="blue")
    plt.plot(rounds, custom_train_acc, label="Bayesian Accuracy", color="red")

    # Líneas horizontales con la precisión final de test
    plt.axhline(
        y=normal_final_test_acc,
        color="blue",
        linestyle="--",
        label="Normal Test Accuracy",
    )
    plt.axhline(
        y=custom_final_test_acc,
        color="red",
        linestyle="--",
        label="Bayesian Test Accuracy",
    )

    plt.xlabel("Boost Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracies")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)

    if output_path:
        plt.savefig(output_path)
    plt.close()
