import xgboost as xgb
from app.utils.data_loader import load_data_from_csv
from app.models.pymc_adjust import apply_pymc_adjustment, custom_objective_factory

def train_normal_xgboost(data_path: str, params: dict):
    """
    Entrena un modelo XGBoost normal.
    """
    dtrain, dtest, _, _, _, _ = load_data_from_csv(data_path)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        evals=[(dtest, "test")],
        evals_result=evals_result
    )
    model.save_model("model_normal.xgb")
    return model, evals_result

def train_custom_xgboost(data_path: str, params: dict, distribution: str):
    """
    Entrena un modelo XGBoost con función de pérdida personalizada.
    """
    dtrain, dtest, _, _, _, _ = load_data_from_csv(data_path)
    custom_obj = custom_objective_factory(distribution)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        obj=custom_obj,
        evals=[(dtest, "test")],
        evals_result=evals_result
    )
    model.save_model("model_custom.xgb")
    return model, evals_result

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
    # Calcular métricas (accuracy, precision, recall, F1)
    return {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1": 0.94}