"""
Data file controller.
Handles API operations related to data files.
"""
import os
from fastapi import APIRouter, HTTPException, File, UploadFile
import aiofiles

from app.core import config
from app.core.logging import data_logger as logger
from app.domain.schemas import (
    FileSelection,
    DataLoadRequest,
    DataPreviewResponse, 
    AvailableFilesResponse,
    MessageResponse
)
from app.modules.data_management.services.file_management import (
    get_data_files,
    select_file,
    load_file,
    upload_file,
    delete_file
)

# Create router
router = APIRouter()


@router.get("/files", response_model=AvailableFilesResponse)
async def get_files():
    """
    Get a list of available data files.
    """
    try:
        files = get_data_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Error getting data files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select", response_model=DataPreviewResponse)
async def select_data_file(selection: FileSelection):
    """
    Select a data file for use in training and prediction.
    """
    try:
        message, preview = select_file(selection.filename)
        return {"message": message, "preview": preview}
    except FileNotFoundError as e:
        logger.error(f"File not found: {selection.filename}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error selecting data file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", response_model=DataPreviewResponse)
async def load_data_file(request: DataLoadRequest):
    """
    Load a data file with specific options.
    """
    try:
        message, preview = load_file(
            request.filename, 
            request.has_header, 
            request.separator
        )
        return {"message": message, "preview": preview}
    except FileNotFoundError:
        logger.error(f"File not found: {request.filename}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DataPreviewResponse)
async def upload_data_file(file: UploadFile = File(...)):
    """
    Upload a new data file.
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Upload file
        message, preview = await upload_file(file.filename, contents)
        
        return {"message": message, "preview": preview}
    except Exception as e:
        logger.error(f"Error uploading data file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")
    finally:
        await file.close()


@router.delete("/delete", response_model=MessageResponse)
async def delete_data_file(selection: FileSelection):
    """
    Delete a data file.
    """
    try:
        message = delete_file(selection.filename)
        return {"message": message}
    except FileNotFoundError:
        logger.error(f"File not found: {selection.filename}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error deleting data file: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred while deleting the file: {e}"
        ) 