from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from app.config import conf_manager
from app.service.xgboost_model import grid_search_xgboost
from app.common.common_methods import get_number_of_classes

router = APIRouter()


class GridSearchResponse(BaseModel):
    best_parameters: dict


class ParameterSelectionRequest(BaseModel):
    parameters: dict


class ParameterSelectionResponse(BaseModel):
    message: str


@router.get("/grid_search", response_model=GridSearchResponse)
async def grid_search():
    try:
        best_params = grid_search_xgboost()
        return {"best_parameters": best_params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    pass


@router.post("/setparams", response_model=ParameterSelectionResponse)
async def select_parameters(request: ParameterSelectionRequest):
    try:
        # Retrieve the loaded data file path from settings
        num_classes = get_number_of_classes()

        # Copy initial parameters from the request
        params = request.parameters.copy()
        params.update({"seed": conf_manager.get_value("kseed")})

        # Update parameters based on the number of classes
        if num_classes > 2:
            params.update(
                {
                    "objective": "multi:softmax",
                    "num_class": num_classes,
                    "eval_metric": "merror",  # Metric for multiclass
                }
            )
        else:
            params.update(
                {
                    "objective": "binary:logistic",
                    "scale_pos_weight": 3,  # Adjustment for class imbalance
                    "eval_metric": "error",  # Metric for binary classification
                }
            )

        # Save parameters to the settings file
        conf_manager.set_value("model_parameters", params)
        return {"message": "Parameters updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
