import os
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException, Form
from models.xgboost_model import evaluate_model

router = APIRouter()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

@router.post("/run")
def test_model(model_path: str = Form(...), data_path: str = Form(...)):
    """
    Evaluates the model, generates result image(s) and returns their paths.
    """
    try:
        metrics = evaluate_model(model_path, data_path)
        
        # Create a bar chart of the evaluation metrics
        fig, ax = plt.subplots()
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        ax.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange'])
        ax.set_title("Evaluation Metrics")
        image_path = os.path.join(RESULTS_DIR, "evaluation_metrics.png")
        plt.savefig(image_path)
        plt.close(fig)
        
        return {"message": "Test completed successfully", "image_paths": [image_path], "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))