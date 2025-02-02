from fastapi import APIRouter, HTTPException
from app.models.xgboost_model import train_normal_xgboost, train_custom_xgboost
from app.schemas.models import TrainRequest, TrainResponse

router = APIRouter()

@router.post("/normal", response_model=TrainResponse)
async def train_normal_model(request: TrainRequest):
    """
    Endpoint para entrenar un modelo XGBoost normal.
    """
    try:
        model, evals_result = train_normal_xgboost(request.data_path, request.params)
        response_data = {"message": "Modelo entrenado exitosamente", "evals_result": evals_result}
        print(response_data)
        print(type(evals_result))
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom", response_model=TrainResponse)
async def train_custom_model(request: TrainRequest):
    """
    Endpoint para entrenar un modelo XGBoost con función de pérdida personalizada.
    """
    try:
        model, evals_result = train_custom_xgboost(request.data_path, request.params, request.distribution)
        return {"message": "Modelo entrenado exitosamente", "evals_result": evals_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))