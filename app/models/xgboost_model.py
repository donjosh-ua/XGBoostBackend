import xgboost as xgb
import pandas as pd
import numpy as np

from utils import conf_manager
from utils.data_loader import load_data_from_csv
from models.pymc_adjust import apply_pymc_adjustment, custom_objective_factory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

kSeed = 42  # constant seed

params = {}

def train_normal_xgboost(data_path: str, params: dict, method: str = "split", num_folds: int = 5):
    """
    Entrena un modelo XGBoost normal usando split o cross validation.
    """
    rounds = conf_manager.get_value("rounds")
    folds = conf_manager.get_value("training_value") if method == "cv" else num_folds

    if method == "split":
        dtrain, dtest, _, _, _, _ = load_data_from_csv()
        evals_result = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result
        )
        model.save_model("app/model_normal.xgb")

        return model, evals_result
    elif method == "cv":
        # Cargar el dataset completo desde CSV
        df = pd.read_csv(data_path)

        if df.shape[1] < 2:
            raise ValueError("El dataset debe tener al menos una columna de características y una etiqueta.")
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        dmat = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(
            params,
            dmat,
            num_boost_round=rounds,
            nfold=folds,
            metrics="logloss",
            seed=kSeed
        )

        return None, cv_results.to_dict()
    else:
        raise ValueError("Método de entrenamiento inválido. Use 'split' o 'cv'.")

def train_custom_xgboost(data_path: str, params: dict, distribution: str, method: str = "split", num_folds: int = 5):
    """
    Entrena un modelo XGBoost con función de pérdida personalizada usando split.
    Cross validation no es soportado para funciones objetivo personalizadas.
    """
    rounds = conf_manager.get_value("rounds")

    if method == "split":

        dtrain, dtest, _, _, _, _ = load_data_from_csv()
        custom_obj = custom_objective_factory(distribution)
        evals_result = {}

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            obj=custom_obj,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result
        )
        model.save_model("app/model_custom.xgb")

        return model, evals_result
    elif method == "cv":
        raise ValueError("El entrenamiento con objetivo personalizado no es soportado en cross validation.")
    else:
        raise ValueError("Método de entrenamiento inválido. Use 'split' o 'cv'.")

def predict_with_model(model_path: str, data_path: str):
    """
    Realiza predicciones con un modelo XGBoost.
    """
    model = xgb.Booster(model_file=model_path)
    dtest = xgb.DMatrix(data_path)

    return model.predict(dtest)

def evaluate_model(model_path: str, data_path: str):
    """
    Evalúa un modelo XGBoost.
    """
    model = xgb.Booster(model_file=model_path)
    dtest = xgb.DMatrix(data_path)
    predictions = model.predict(dtest)

    # Calcular métricas (accuracy, precision, recall, F1) aquí se retornan valores fijos como ejemplo
    return {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1": 0.94}

def cross_validate(X: np.ndarray, y: np.ndarray, params: dict, n_splits: int = 5, is_multiclass: bool = False, distribution: str = None):
    """
    Realiza validación cruzada usando el modelo XGBoost normal.
    Si se especifica una distribución se lanza un error ya que la validación cruzada con objetivo personalizado no es soportada.
    Calcula métricas de accuracy, precision, recall y F1 para cada fold.
    """
    if distribution is not None:
        raise ValueError("El entrenamiento con objetivo personalizado no es soportado en cross_validation.")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kSeed)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear matrices de XGBoost
        dtrain_fold = xgb.DMatrix(X_train, label=y_train)
        dtest_fold = xgb.DMatrix(X_test, label=y_test)

        # Entrenar modelo normal
        evals_result = {}
        booster = xgb.train(params, dtrain_fold, num_boost_round=50, evals=[(dtest_fold, "test")], evals_result=evals_result)

        # Realizar predicciones
        preds = booster.predict(dtest_fold)
        # Para binario se aplica un umbral de 0.5
        pred_labels = (preds >= 0.5).astype(int)

        # Calcular métricas
        accuracies.append(accuracy_score(y_test, pred_labels))
        precisions.append(precision_score(y_test, pred_labels, zero_division=0))
        recalls.append(recall_score(y_test, pred_labels, zero_division=0))
        f1s.append(f1_score(y_test, pred_labels, zero_division=0))

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s)
    }


# def train_normal_xgboost(data_path: str, params: dict):
#     """
#     Entrena un modelo XGBoost normal.
#     """
#     dtrain, dtest, _, _, _, _ = load_data_from_csv(data_path)
#     evals_result = {}
#     model = xgb.train(
#         params,
#         dtrain,
#         num_boost_round=50,
#         evals=[(dtest, "test")],
#         evals_result=evals_result
#     )
#     model.save_model("model_normal.xgb")
#     return model, evals_result

# def train_custom_xgboost(data_path: str, params: dict, distribution: str):
#     """
#     Entrena un modelo XGBoost con función de pérdida personalizada.
#     """
#     dtrain, dtest, _, _, _, _ = load_data_from_csv(data_path)
#     custom_obj = custom_objective_factory(distribution)
#     evals_result = {}
#     model = xgb.train(
#         params,
#         dtrain,
#         num_boost_round=50,
#         obj=custom_obj,
#         evals=[(dtest, "test")],
#         evals_result=evals_result
#     )
#     model.save_model("model_custom.xgb")
#     return model, evals_result

# def predict_with_model(model_path: str, data_path: str):
#     """
#     Realiza predicciones con un modelo XGBoost.
#     """
#     model = xgb.Booster(model_file=model_path)
#     dtest = xgb.DMatrix(data_path)
#     return model.predict(dtest)

# def evaluate_model(model_path: str, data_path: str):
#     """
#     Evalúa un modelo XGBoost.
#     """
#     model = xgb.Booster(model_file=model_path)
#     dtest = xgb.DMatrix(data_path)
#     predictions = model.predict(dtest)
#     # Calcular métricas (accuracy, precision, recall, F1)
#     return {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1": 0.94}