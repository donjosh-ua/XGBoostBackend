"""
Main application entry point.
Sets up and configures the FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.logging import app_logger as logger

# Create FastAPI app
app = FastAPI(
    title="XGBoost with FastAPI",
    description="API for training and predicting with XGBoost models and neural networks",
    version="1.0.0",
)

# Configure CORS
origins = [
    "http://localhost:5173",  # frontend url
    "https://xgboostfrontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(router)


@app.get("/", tags=["root"])
def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Bienvenido a la API de XGBoost con FastAPI"}


@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts.
    Initialize resources, connections, etc.
    """
    logger.info("Application started")
    
    # Ensure required directories exist
    from app.common.utils import ensure_directory_exists
    ensure_directory_exists("app/data/datasets")
    ensure_directory_exists("app/data/models/xgboost")
    ensure_directory_exists("app/data/models/neural_network")
    ensure_directory_exists("app/data/outputs/plots")
    ensure_directory_exists("app/data/outputs/results")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the application shuts down.
    Clean up resources, close connections, etc.
    """
    logger.info("Application shutting down")


# Run the application using Uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
