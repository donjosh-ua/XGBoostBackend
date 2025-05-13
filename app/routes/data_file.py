import os
import pandas as pd
import aiofiles
from pydantic import BaseModel
from app.utils import conf_manager
from app.utils.common_methods import get_number_of_classes
from fastapi import APIRouter, HTTPException, File, UploadFile


router = APIRouter()

DATA_DIR = os.path.join(".", "app", "data")


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
        with open(file_path, "r") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                preview.append(line.rstrip("\n"))
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
            if "scale_pos_weight" in params:
                del params["scale_pos_weight"]
            # Update for multiclass
            params.update(
                {
                    "objective": "multi:softmax",
                    "num_class": num_classes,
                    "eval_metric": "merror",  # Metric for multiclass
                }
            )
        else:
            # Remove num_class if exists
            if "num_class" in params:
                del params["num_class"]
            # Update for binary classification
            params.update(
                {
                    "objective": "binary:logistic",
                    "scale_pos_weight": 3,  # Adjustment for class imbalance
                    "eval_metric": "error",  # Metric for binary classification
                }
            )

        # Save updated parameters to the settings file
        conf_manager.set_value("model_parameters", params)

        preview = df.head(10).to_dict(orient="records")
        return {
            "message": f"File {request.filename} loaded successfully",
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload(file: UploadFile = File(...), separator: str = ","):

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    # Ensure DATA_DIR exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        contents = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong: " + str(e))
    finally:
        await file.close()

    # After upload, load the file and update the configuration
    try:
        df = pd.read_csv(file_path, sep=separator)
        preview = df.head(10).to_dict(orient="records")

        # Update the config with the new file settings
        conf_manager.set_value("selected_file", safe_filename)
        conf_manager.set_value("loaded_data_path", file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload succeeded, but file could not be loaded: {e}",
        )

    return {
        "message": f"Successfully uploaded and loaded {safe_filename}",
        "preview": preview,
    }


@router.delete("/delete")
def delete_data_file(selection: FileSelection):

    safe_filename = os.path.basename(selection.filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        # Optionally, update the configuration if the deleted file was selected/loaded
        current_selected = conf_manager.get_value("selected_file")
        if current_selected == safe_filename:
            conf_manager.set_value("selected_file", "")
            conf_manager.set_value("loaded_data_path", "")
        return {"message": f"File {safe_filename} has been deleted."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred while deleting the file: {e}"
        )
