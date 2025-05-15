from collections import OrderedDict

from pydantic import BaseModel
from typing import Dict, List, Optional


class TrainRequest(BaseModel):
    data_path: str
    params: Dict[str, str]
    distribution: Optional[str] = "Normal"


class TrainResponse(BaseModel):
    # message: str
    # evals_result: Dict[str, List[float]]
    message: str
    # evals_result: Dict[str, OrderedDict[str, List[float]]]  # Corrected type


class PredictRequest(BaseModel):
    model_path: str
    data_path: str


class PredictResponse(BaseModel):
    predictions: List[float]


class EvaluateRequest(BaseModel):
    model_path: str
    data_path: str


class EvaluateResponse(BaseModel):
    metrics: Dict[str, float]
