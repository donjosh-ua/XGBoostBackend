import os
import numpy as np
import xgboost as xgb
from fastapi import APIRouter, HTTPException
from utils.common_methods import *
from utils.data_loader import load_data_from_csv

router = APIRouter()

@router.post("/run")
def test_models_plots():
    """
    Evaluates both normal and custom XGBoost models, generates evaluation plots,
    label distributions and confusion matrices and saves them in the data/plots folder.
    """
    # Compute plots folder (create if not exists)
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    plots_dir = os.path.join(BASE_DIR, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    num_classes = get_number_of_classes()
    is_multiclass = num_classes > 2

    print("Number of classes:", num_classes)

    # Load data from CSV
    dtrain, dtest, train_x, train_y, test_x, test_y = load_data_from_csv()

    # Load models from saved files
    normal_model_path = os.path.join("app", "model_normal.xgb")
    custom_model_path = os.path.join("app", "model_custom.xgb")
    if not os.path.isfile(normal_model_path) or not os.path.isfile(custom_model_path):
        raise HTTPException(status_code=400, detail="Model files not found. Please train the models first.")

    normal_model = xgb.Booster()
    normal_model.load_model(normal_model_path)
    custom_model = xgb.Booster()
    custom_model.load_model(custom_model_path)

    # Generate predictions
    if is_multiclass:
        normal_preds = normal_model.predict(dtest)
        custom_preds = custom_model.predict(dtest)
    else:
        normal_preds = np.round(normal_model.predict(dtest))
        custom_preds = np.round(custom_model.predict(dtest))

    # Save plots instead of showing them.
    normal_metrics_path = os.path.join(plots_dir, "display_metrics_normal.png")
    custom_metrics_path = os.path.join(plots_dir, "display_metrics_custom.png")
    distribution_path = os.path.join(plots_dir, "label_distribution_side_by_side.png")
    confusion_path = os.path.join(plots_dir, "confusion_matrices.png")

    display_metrics(normal_preds, test_y, is_multiclass, title="Normal XGBoost", output_path=normal_metrics_path)
    display_metrics(custom_preds, test_y, is_multiclass, title="Bayesian Objective", output_path=custom_metrics_path)
    plot_label_distributions_side_by_side(test_y, normal_preds, custom_preds, output_path=distribution_path)
    show_confusion_matrices_side_by_side(test_y, normal_preds, custom_preds, output_path=confusion_path)

    return {
        "message": "Testing plots generated and saved successfully",
        "plots_folder": plots_dir
    }
