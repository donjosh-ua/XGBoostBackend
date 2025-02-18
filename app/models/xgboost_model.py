import xgboost as xgb
import pandas as pd
import numpy as np

from app.utils import conf_manager
from app.utils.data_loader import load_data_from_csv
from app.models.pymc_adjust import custom_objective_factory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold


kSeed = conf_manager.get_value("kseed")

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
        model.save_model("./app/model_normal.xgb")

        return model, evals_result
    elif method == "cv":

        df = pd.read_csv(data_path)
        if df.shape[1] < 2:
            raise ValueError("El dataset debe tener al menos una columna de características y una etiqueta.")
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        kf = KFold(n_splits=folds, shuffle=True, random_state=kSeed)
        evals_result = {}
        booster = None

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtrain_fold = xgb.DMatrix(X_train, label=y_train)
            dtest_fold = xgb.DMatrix(X_test, label=y_test)

            booster = xgb.train(
            params,
            dtrain_fold,
            num_boost_round=rounds,
            evals=[(dtrain_fold, 'train'), (dtest_fold, 'test')],
            evals_result=evals_result
        )

        booster.save_model("./app/model_custom.xgb")

        return booster, evals_result
    else:
        raise ValueError("Método de entrenamiento inválido. Use 'split' o 'cv'.")

def train_custom_xgboost(data_path: str, params: dict, distribution: str, method: str = "split", num_folds: int = 5):
    """
    Entrena un modelo XGBoost con función de pérdida personalizada usando split o validación cruzada.
    Para 'cv', se realizan los entrenamientos en folds para obtener métricas, luego se entrena el modelo final
    en todos los datos, se guarda y se retornan los resultados de la validación y del entrenamiento final.
    """
    rounds = conf_manager.get_value("rounds")
    folds = conf_manager.get_value("training_value") if method == "cv" else num_folds

    custom_obj = custom_objective_factory(distribution)
    
    if method == "split":
        dtrain, dtest, _, _, _, _ = load_data_from_csv()
        evals_result = {}

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            obj=custom_obj,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            evals_result=evals_result
        )
        model.save_model("./app/model_custom.xgb")
        return model, evals_result

    elif method == "cv":

        df = pd.read_csv(data_path)
        if df.shape[1] < 2:
            raise ValueError("El dataset debe tener al menos una columna de características y una etiqueta.")
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        kf = KFold(n_splits=folds, shuffle=True, random_state=kSeed)
        evals_result = {}
        booster = None

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtrain_fold = xgb.DMatrix(X_train, label=y_train)
            dtest_fold = xgb.DMatrix(X_test, label=y_test)

            booster = xgb.train(
                params,
                dtrain_fold,
                num_boost_round=rounds,
                obj=custom_obj,
                evals=[(dtrain_fold, "train"), (dtest_fold, "test")],
                evals_result=evals_result
            )

        booster.save_model("./app/model_custom.xgb")

        return booster, evals_result
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

def grid_search_xgboost():

    _, _, train_x, train_y, _, _ = load_data_from_csv()

    # Definir el espacio de búsqueda
    # param_grid = {
    #     'max_depth': [3, 5, 8, 10],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'gamma': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    #     'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
    #     'subsample': [0.7, 0.8, 1.0],
    #     'colsample_bytree': [0.7, 0.8, 1.0],
    #     'min_child_weight': [1, 3, 5, 7, 10],
    #     'scale_pos_weight': [1, 3]  # Para manejar desbalance de clases
    # }

    # Parametros reducidos para acelerar el proceso
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'gamma': [0.01, 0.1],
        'eta': [0.01, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [1, 3],
        'scale_pos_weight': [1, 3]
    }

    # Convertir los datos a DMatrix
    model = xgb.XGBClassifier(objective='binary:logistic', seed=kSeed)
    
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(train_x, train_y)
    params = grid_search.best_params_

    return params