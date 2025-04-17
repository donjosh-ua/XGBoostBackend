import logging
import pymc as pm
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Constantes
kSeed = 1994
kRatio = 0.7
num_rounds = 50
np.random.seed(kSeed)

# Archivo de datos
datafile = 'pima-indians-diabetes.data.csv'
# datafile = 'glass.csv'

# Configuracion de la distribución
distribution = 'Normal'
distribution_params = {
    'Normal': {'mu': 0, 'sigma': 10},
    'HalfNormal': {'sigma': 10},
    'Cauchy': {'alpha': 0, 'beta': 1},
    'Exponential': {'lam': 1}
}

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config_params(distribution: str):
    params = distribution_params[distribution]
    return params


def load_data_from_csv() -> tuple:
    """
    Cargar datos desde el archivo CSV y dividirlos en entrenamiento y prueba.
    """
    data = pd.read_csv(datafile, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=1 - kRatio, random_state=kSeed
    )

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    return dtrain, dtest, train_x, train_y, test_x, test_y


def optimize_xgboost(dtrain, dtest):
    param_grid = {
        'max_depth': [3, 5, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'gamma': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10],
        'scale_pos_weight': [1, 3]  # Para manejar desbalance de clases
    }

    # Convertir los datos a DMatrix
    X_train, y_train = dtrain
    model = xgb.XGBClassifier(objective='binary:logistic', seed=kSeed)

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def apply_pymc_adjustment(preds: np.ndarray) -> np.ndarray:
    """
    Ajusta las predicciones con PyMC de forma genérica usando la distribución seleccionada.
    Soporta binario (1D) o multiclass (2D) aplicando logistic o softmax.
    """
    preds = np.array(preds)
    params = get_config_params(distribution)

    # Mapeo de distribuciones
    distributions = {
        'Normal': lambda name, params, shape: pm.Normal(
            name, mu=params['mu'], sigma=params['sigma'], shape=shape
        ),
        'HalfNormal': lambda name, params, shape: pm.HalfNormal(
            name, sigma=params['sigma'], shape=shape
        ),
        'Cauchy': lambda name, params, shape: pm.Cauchy(
            name, alpha=params['alpha'], beta=params['beta'], shape=shape
        ),
        'Exponential': lambda name, params, shape: pm.Exponential(
            name, lam=params['lam'], shape=shape
        )
    }

    # Detección de binario o multiclass
    is_multiclass = (preds.ndim > 1)
    shape = preds.shape[1] if is_multiclass else None

    dist_func = distributions.get(distribution)
    if dist_func is None:
        logger.error(f"Distribución '{distribution}' no soportada.")
        raise ValueError(f"Distribución '{distribution}' no soportada.")
    with pm.Model() as model:
        adjustment = dist_func("adjustment", params, shape=shape)
        adjusted_logits = preds + adjustment

        trace = pm.sample(1000, tune=1000, chains=4, cores=4, progressbar=True)
        # trace = pm.sample(1000, return_inferencedata=False, progressbar=False)

        if is_multiclass:
            adjustment_samples = trace.posterior['adjustment'].mean(dim=("chain", "draw")).values
            adjusted_logits = preds + adjustment_samples
            exp_vals = np.exp(adjusted_logits)
            adjusted_preds = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            adjustment_samples = trace.posterior['adjustment'].mean().values
            adjusted_logits = preds + adjustment_samples
            adjusted_preds = expit(adjusted_logits)

    return adjusted_preds


def custom_objective_factory(distribution: str):
    """
    Crea una función de objetivo personalizada con la distribución especificada.
    """
    def custom_objective(preds, dtrain):
        adjusted_preds = apply_pymc_adjustment(preds)
        labels = dtrain.get_label()

        if preds.ndim == 1:
            # Binario
            grad = adjusted_preds - labels
            hess = adjusted_preds * (1 - adjusted_preds)
        else:
            # Multiclass (aprox. de grad/hess)
            rows = len(labels)
            grad = np.copy(adjusted_preds)
            grad[np.arange(rows), labels.astype(int)] -= 1.0
            hess = adjusted_preds * (1.0 - adjusted_preds)
        return grad, hess
    return custom_objective


def train_normal_xgboost(dtrain, dtest, params, num_boost_round=num_rounds):
    """
    Entrena un modelo XGBoost normal (sin función personalizada).
    Retorna el modelo y el evals_result para poder trazar las curvas de entrenamiento.
    """
    evals_result = {}
    model_normal = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result
    )
    return model_normal, evals_result


def train_custom_xgboost(dtrain, dtest, params, distribution: str, num_boost_round=num_rounds):
    """
    Entrena un modelo XGBoost usando la función de pérdida personalizada con distribución seleccionada.
    Retorna el modelo y el evals_result para poder trazar las curvas de entrenamiento.
    """
    evals_result = {}
    custom_obj = custom_objective_factory(distribution)
    model_custom = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=custom_obj,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result
    )
    return model_custom, evals_result


def get_number_of_classes():
    """Obtener el número de clases."""
    data = pd.read_csv(datafile, header=None)
    return len(data.iloc[:, -1].unique())


def    display_metrics(preds: np.ndarray, test_y: np.ndarray, is_multiclass: bool, title: str = "Métricas de Rendimiento"):
    """Mostrar métricas de rendimiento en una figura destacada."""
    metrics = {
        'Accuracy': accuracy_score(test_y, preds),
        'Precision': precision_score(test_y, preds, average='weighted' if is_multiclass else 'binary'),
        'Recall': recall_score(test_y, preds, average='weighted' if is_multiclass else 'binary'),
        'F1 Score': f1_score(test_y, preds, average='weighted' if is_multiclass else 'binary')
    }

    # Crear una figura con una tabla
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table = ax.table(cellText=[[metric, f"{value:.4f}"] for metric, value in metrics.items()],
                     colLabels=['Métrica', 'Valor'],
                     loc='center',
                     cellLoc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.title(title, fontsize=16)
    plt.show()


def plot_roc_curve(test_y, preds):
    fpr, tpr, thresholds = roc_curve(test_y, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_label_distributions(true_labels: np.ndarray, predicted_labels: np.ndarray, is_multiclass: bool):
    """Plot the distribution of true labels vs. predicted labels."""
    unique_classes = np.unique(true_labels)
    true_counts = [np.sum(true_labels == cls) for cls in unique_classes]
    pred_counts = [np.sum(predicted_labels == cls) for cls in unique_classes]

    x_positions = np.arange(len(unique_classes))
    width = 0.4

    plt.figure()
    plt.bar(x_positions - width/2, true_counts, width=width, label='True Labels', color='blue')
    plt.bar(x_positions + width/2, pred_counts, width=width, label='Predicted Labels', color='red')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('True vs Predicted Label Distribution')
    plt.xticks(x_positions, unique_classes)
    plt.legend()
    plt.show()


def show_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, is_multiclass: bool):
    """Plot the confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


def show_confusion_matrices_side_by_side(true_labels: np.ndarray,
                                         normal_preds: np.ndarray,
                                         adjusted_preds: np.ndarray):
    """Plot two confusion matrices side by side: Normal vs. PyMC-Adjusted."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normal XGBoost
    cm_normal = confusion_matrix(true_labels, normal_preds)
    disp_normal = ConfusionMatrixDisplay(confusion_matrix=cm_normal)
    disp_normal.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title("Normal XGBoost")

    # PyMC-Adjusted
    cm_adjusted = confusion_matrix(true_labels, adjusted_preds)
    disp_adjusted = ConfusionMatrixDisplay(confusion_matrix=cm_adjusted)
    disp_adjusted.plot(ax=axes[1], cmap=plt.cm.Blues)
    axes[1].set_title("PyMC-Adjusted")

    plt.tight_layout()
    plt.show()


def plot_label_distributions_side_by_side(true_labels: np.ndarray,
                                          normal_preds: np.ndarray,
                                          adjusted_preds: np.ndarray):
    """Plot normal vs. adjusted label distribution side by side."""
    unique_classes = np.unique(true_labels)
    true_counts = [np.sum(true_labels == cls) for cls in unique_classes]
    normal_counts = [np.sum(normal_preds == cls) for cls in unique_classes]
    adjusted_counts = [np.sum(adjusted_preds == cls) for cls in unique_classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normal
    x_positions = np.arange(len(unique_classes))
    width = 0.3
    axes[0].bar(x_positions - width/2, true_counts, width=width, label='True', color='blue')
    axes[0].bar(x_positions + width/2, normal_counts, width=width, label='Normal', color='red')
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(unique_classes)
    axes[0].set_title('Normal XGBoost Distribution')
    axes[0].legend()

    # Adjusted
    axes[1].bar(x_positions - width/2, true_counts, width=width, label='True', color='blue')
    axes[1].bar(x_positions + width/2, adjusted_counts, width=width, label='Adjusted', color='red')
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(unique_classes)
    axes[1].set_title('PyMC-Adjusted Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_accuracy_lines_and_curves(normal_evals_result, custom_evals_result, is_multiclass):
    """
    Muestra dos curvas de precisión de entrenamiento (Normal y Bayesiano)
    y dos líneas horizontales con la precisión final de test (Normal y Bayesiano).
    """
    metric = 'merror' if is_multiclass else 'error'

    # Entrenamiento
    normal_train_err = normal_evals_result['train'][metric]
    custom_train_err = custom_evals_result['train'][metric]
    normal_train_acc = [1.0 - e for e in normal_train_err]
    custom_train_acc = [1.0 - e for e in custom_train_err]

    # Test (para representar sólo la versión final en una línea)
    normal_test_err = normal_evals_result['test'][metric]
    custom_test_err = custom_evals_result['test'][metric]
    normal_final_test_acc = 1.0 - normal_test_err[-1]
    custom_final_test_acc = 1.0 - custom_test_err[-1]

    rounds = range(len(normal_train_acc))

    plt.figure(figsize=(8, 6))

    # Curvas de training accuracy
    plt.plot(rounds, normal_train_acc, label='Normal Accuracy', color='blue')
    plt.plot(rounds, custom_train_acc, label='Bayesian Accuracy', color='red')

    # Líneas horizontales con la precisión final de test
    plt.axhline(y=normal_final_test_acc, color='blue', linestyle='--', label='Normal Test Accuracy')
    plt.axhline(y=custom_final_test_acc, color='red', linestyle='--', label='Bayesian Test Accuracy')

    plt.xlabel('Boost Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()


def get_xgb_params(num_classes):
    params = {
        'seed': kSeed,
        'eta': 0.05,
        'max_depth': 5,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 1.0
    }

    if num_classes > 2:
        params.update({
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eval_metric': 'merror'  # Métrica para multiclass
        })
    else:
        params.update({
            'objective': 'binary:logistic',
            'scale_pos_weight': 3,  # Ajuste para desbalance de clases
            'eval_metric': 'error'  # Métrica para binario
        })

    return params

def cross_validate(X: np.ndarray, y: np.ndarray, params: dict, n_splits: int = 5, is_multiclass: bool = False):
    """
    Realiza validación cruzada usando el modelo XGBoost con la función de pérdida personalizada.
    Muestra métricas de exactitud, precisión, recall y F1 en formato de tabla.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kSeed)

    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear matrices de XGBoost
        dtrain_fold = xgb.DMatrix(X_train, label=y_train)
        dtest_fold = xgb.DMatrix(X_test, label=y_test)

        # Entrenar con la función de pérdida personalizada
        custom_booster, _ = train_custom_xgboost(dtrain_fold, dtest_fold, params, distribution)

        # Generar predicciones
        preds = custom_booster.predict(dtest_fold)
        if not is_multiclass:
            preds = np.round(preds)

        # Calcular métricas
        accuracies.append(accuracy_score(y_test, preds))
        precisions.append(precision_score(y_test, preds, average='weighted' if is_multiclass else 'binary'))
        recalls.append(recall_score(y_test, preds, average='weighted' if is_multiclass else 'binary'))
        f1s.append(f1_score(y_test, preds, average='weighted' if is_multiclass else 'binary'))

    # Construir DataFrame con las métricas promedio
    metrics_mean = {
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Valor': [
            f"{np.mean(accuracies):.4f}",
            f"{np.mean(precisions):.4f}",
            f"{np.mean(recalls):.4f}",
            f"{np.mean(f1s):.4f}"
        ]
    }

    df_metrics = pd.DataFrame(metrics_mean)

    # Mostrar métricas en una figura con tabla
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table = ax.table(cellText=df_metrics.values,
                     colLabels=df_metrics.columns,
                     loc='center',
                     cellLoc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.title('Cross-Validation', fontsize=16)
    plt.show()

def main():
    """
    Flujo principal del programa para entrenar XGBoost con y sin función de pérdida personalizada,
    mostrar métricas, distribuciones de etiquetas, matrices de confusión y gráficas de exactitud.
    """

    num_classes = get_number_of_classes()
    is_multiclass = num_classes > 2
    params = get_xgb_params(num_classes)

    # Cargar datos
    dtrain, dtest, train_x, train_y, test_x, test_y = load_data_from_csv()

    # Entrenar xgboost normal
    logger.info("Entrenando modelo XGBoost normal...")
    normal_model, normal_evals = train_normal_xgboost(dtrain, dtest, params)
    if is_multiclass:
        normal_preds = normal_model.predict(dtest)
    else:
        normal_preds = np.round(normal_model.predict(dtest))

    # Entrenar xgboost con función personalizada

    logger.info("Entrenando modelo XGBoost con función bayesiana...")
    custom_model, custom_evals = train_custom_xgboost(dtrain, dtest, params, distribution)
    if is_multiclass:
        custom_preds = custom_model.predict(dtest)
    else:
        custom_preds = np.round(custom_model.predict(dtest))

    # Mostrar métricas
    # logger.info("==== Normal XGBoost ====")
    display_metrics(normal_preds, test_y, is_multiclass, title="Normal XGBoost")
    # logger.info("==== Bayesian Objective ====")
    display_metrics(custom_preds, test_y, is_multiclass, title="Bayesian Objective")

    # Distribuciones y confusiones
    plot_label_distributions(test_y, normal_preds, is_multiclass)
    plot_label_distributions(test_y, custom_preds, is_multiclass)
    show_confusion_matrix(test_y, normal_preds, is_multiclass)
    show_confusion_matrix(test_y, custom_preds, is_multiclass)

    # Ahora mostrar lado a lado
    plot_label_distributions_side_by_side(test_y, normal_preds, custom_preds)
    show_confusion_matrices_side_by_side(test_y, normal_preds, custom_preds)

    # Llamamos a la función para mostrar la curva de entrenamiento
    plot_accuracy_lines_and_curves(normal_evals, custom_evals, is_multiclass)

    # Cross Validation. Puede tardar un poco.
    cross_validate(train_x, train_y, params, n_splits=5, is_multiclass=is_multiclass)

if __name__ == "__main__":
    main()
