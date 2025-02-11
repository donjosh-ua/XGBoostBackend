import os
import pandas as pd
from pydantic import BaseModel
from utils import conf_manager 
from fastapi import APIRouter, HTTPException
from utils.common_methods import get_number_of_classes

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class FileSelection(BaseModel):
    filename: str

class DataLoadRequest(BaseModel):
    filename: str
    has_header: bool
    separator: str

@router.get("/files")
def get_data_files():
    try:
        files = os.listdir(DATA_DIR)
        csv_files = [f for f in files if f.endswith(".csv")]
        return {"files": csv_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select")
def select_data_file(selection: FileSelection):
    file_path = os.path.join(DATA_DIR, selection.filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Update global config file
    conf_manager.set_value("selected_file", selection.filename)
    
    preview = []
    try:
        with open(file_path, 'r') as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                preview.append(line.rstrip('\n'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Selected file set to {selection.filename}", "preview": preview}

@router.post("/load")
def load_data_file(request: DataLoadRequest):
    file_path = os.path.join(DATA_DIR, request.filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    # Save the selected file path in the config file
    conf_manager.set_value("selected_file", request.filename)
        
    header = 0 if request.has_header else None
    try:
        df = pd.read_csv(file_path, sep=request.separator, header=header)
        # Persist additional settings: file path, header info, and separator used.
        conf_manager.set_value("loaded_data_path", file_path)
        conf_manager.set_value("has_header", request.has_header)
        conf_manager.set_value("separator", request.separator)

        # Retrieve the loaded data file path from settings
        num_classes = get_number_of_classes()

        # Copy initial parameters from the settings
        params = conf_manager.get_value("model_parameters")

        # Update parameters based on the number of classes with property adjustments
        if num_classes > 2:
            # Remove scale_pos_weight if exists
            if 'scale_pos_weight' in params:
                del params['scale_pos_weight']
            # Update for multiclass
            params.update({
                'objective': 'multi:softmax',
                'num_class': num_classes,
                'eval_metric': 'merror'  # Metric for multiclass
            })
        else:
            # Remove num_class if exists
            if 'num_class' in params:
                del params['num_class']
            # Update for binary classification
            params.update({
                'objective': 'binary:logistic',
                'scale_pos_weight': 3,  # Adjustment for class imbalance
                'eval_metric': 'error'  # Metric for binary classification
            })

        # Save updated parameters to the settings file
        conf_manager.set_value("model_parameters", params)

        preview = df.head(10).to_dict(orient="records")
        return {"message": f"File {request.filename} loaded successfully", "preview": preview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
