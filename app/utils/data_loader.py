import pandas as pd
import xgboost as xgb
from app.utils import conf_manager
from sklearn.model_selection import train_test_split

def load_data_from_csv() -> tuple:
    """
    Load data from the CSV file specified in the settings file and split it into training and test sets.
    The settings file should provide:
      - "loaded_data_path": full path to the CSV file.
      - "train_ratio": a float (e.g. 0.7 for 70% training data).
      - "seed": a random seed for reproducibility.
    If any of these values are not set, defaults are used.
    """
    datafile = conf_manager.get_value("loaded_data_path")
    if not datafile:
        print("No data file loaded. Please load a data file first using the /load endpoint.")
    
    # Load train_ratio and seed from settings, with defaults if not set.
    train_ratio = conf_manager.get_value("training_value")
    if train_ratio is None or train_ratio >= 1:
        train_ratio = 0.7  # default training ratio
    
    header = conf_manager.get_value("has_header")
    header = 0 if header else None 

    seed = conf_manager.get_value("kseed")  # default random seed

    data = pd.read_csv(datafile, header=header, sep=conf_manager.get_value("separator"))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed
    )

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    return dtrain, dtest, train_x, train_y, test_x, test_y


def gen_test_data():

    datafile = conf_manager.get_value("loaded_data_path")
    if not datafile:
        print("No data file loaded. Please load a data file first using the /load endpoint.")
    
    # Load train_ratio and seed from settings, with defaults if not set.
    train_ratio = 0.7  # default training ratio
    
    header = conf_manager.get_value("has_header")
    if header is not None:
        header = 0  # default header

    seed = conf_manager.get_value("kseed")

    data = pd.read_csv(datafile, header=header, sep=conf_manager.get_value("separator"))
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed
    )

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    return dtrain, dtest, train_x, train_y, test_x, test_y