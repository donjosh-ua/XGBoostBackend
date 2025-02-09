from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.xgboost_model import set_model_parameters

router = APIRouter()

class GridSearchRequest(BaseModel):
    model_path: str
    data_path: str

class GridSearchResponse(BaseModel):
    best_parameters: dict

class ParameterSelectionRequest(BaseModel):
    parameters: dict

class ParameterSelectionResponse(BaseModel):
    message: str

@router.post("/grid_search", response_model=GridSearchResponse)
async def grid_search(request: GridSearchRequest):
    # try:
    #     best_params = grid_search_xgboost(request.model_path, request.data_path)
    #     return {"best_parameters": best_params}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    pass

@router.post("/setparams", response_model=ParameterSelectionResponse)
async def select_parameters(request: ParameterSelectionRequest):
    try:
        # updated_params = set_model_parameters(request.parameters)
        set_model_parameters(request.parameters)
        return {"message": "Parameters updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))