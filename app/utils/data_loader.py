import pandas as pd
import xgboost as xgb

def load_data_from_csv(data_path: str):
    """
    Carga datos desde un archivo CSV y los convierte en DMatrix.
    """
    data = pd.read_csv(data_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    dtrain = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X, label=y)  # Usamos los mismos datos para prueba
    return dtrain, dtest, X, y, X, y