import os
import base64
import numpy as np
import xgboost as xgb
from app.utils.common_methods import *
from app.utils.data_loader import load_data_from_csv
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/run")
def test_models_plots():
    """
    Evaluates both normal and custom XGBoost models, generates evaluation plots,
    label distributions and confusion matrices, saves them in the data/plots folder,
    and returns the base64 encoded images so that the frontend can render and store them.
    """
    # Compute plots folder (create if not exists)
    plots_dir = os.path.join("app", "data", "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    num_classes = get_number_of_classes()
    is_multiclass = num_classes > 2

    _, dtest, _, _, _, test_y = load_data_from_csv()

    normal_model_path = os.path.join("app", "model_normal.xgb")
    custom_model_path = os.path.join("app", "model_custom.xgb")
    if not os.path.isfile(normal_model_path) or not os.path.isfile(custom_model_path):
        raise HTTPException(
            status_code=400,
            detail="Model files not found. Please train the models first.",
        )

    normal_model = xgb.Booster()
    normal_model.load_model(normal_model_path)
    custom_model = xgb.Booster()
    custom_model.load_model(custom_model_path)

    if is_multiclass:
        normal_preds = normal_model.predict(dtest)
        custom_preds = custom_model.predict(dtest)
    else:
        normal_preds = np.round(normal_model.predict(dtest))
        custom_preds = np.round(custom_model.predict(dtest))

    normal_metrics_path = os.path.join(plots_dir, "display_metrics_normal.png")
    custom_metrics_path = os.path.join(plots_dir, "display_metrics_custom.png")
    distribution_path = os.path.join(plots_dir, "label_distribution_side_by_side.png")
    confusion_path = os.path.join(plots_dir, "confusion_matrices.png")
    accuracies_path = os.path.join(plots_dir, "accuracies.png")

    display_metrics(
        normal_preds,
        test_y,
        is_multiclass,
        title="Normal XGBoost",
        output_path=normal_metrics_path,
    )
    display_metrics(
        custom_preds,
        test_y,
        is_multiclass,
        title="Bayesian Objective",
        output_path=custom_metrics_path,
    )
    plot_label_distributions_side_by_side(
        test_y, normal_preds, custom_preds, output_path=distribution_path
    )
    show_confusion_matrices_side_by_side(
        test_y, normal_preds, custom_preds, output_path=confusion_path
    )

    def load_image_as_base64(file_path: str) -> str:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Convert saved images to base64 encoded strings.
    normal_metrics_b64 = load_image_as_base64(normal_metrics_path)
    custom_metrics_b64 = load_image_as_base64(custom_metrics_path)
    distribution_b64 = load_image_as_base64(distribution_path)
    confusion_b64 = load_image_as_base64(confusion_path)
    accuracies_b64 = load_image_as_base64(accuracies_path)

    return {
        "message": "Testing plots generated, saved, and encoded successfully",
        "images": {
            "metrics_normal": normal_metrics_b64,
            "metrics_custom": custom_metrics_b64,
            "distribution": distribution_b64,
            "confusion": confusion_b64,
            "accuracies": accuracies_b64,
        },
    }
