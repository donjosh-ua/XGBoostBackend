from fastapi import APIRouter, HTTPException
from app.models.xgboost_model import predict_with_model, evaluate_model
from app.schemas.models import PredictRequest, PredictResponse, EvaluateRequest, EvaluateResponse

router = APIRouter()

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Endpoint para realizar predicciones con un modelo XGBoost.
    """
    try:
        predictions = predict_with_model(request.model_path, request.data_path)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Endpoint para evaluar un modelo XGBoost.
    """
    try:
        metrics = evaluate_model(request.model_path, request.data_path)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))