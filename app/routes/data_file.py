import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils import conf_manager 

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
        # Optionally, store additional info in the config; for large data use other mechanisms
        conf_manager.set_value("loaded_data_path", file_path)
        preview = df.head(10).to_dict(orient="records")
        return {"message": f"File {request.filename} loaded successfully", "preview": preview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))