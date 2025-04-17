from fastapi import FastAPI
from app.controller import training
from fastapi.middleware.cors import CORSMiddleware
from app.controller import predict, data_file, tunning, testing, neural_network

app = FastAPI(
    title="XGBoost with FastAPI",
    description="API para entrenar y predecir con modelos XGBoost y ajuste bayesiano.",
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

# Incluir routers
app.include_router(data_file.router, prefix="/data", tags=["Data"])
app.include_router(tunning.router, prefix="/parameters", tags=["Parámetros"])
app.include_router(training.router, prefix="/train", tags=["Entrenamiento"])
app.include_router(testing.router, prefix="/test", tags=["Testing"])
app.include_router(predict.router, prefix="/predict", tags=["Predicción"])
app.include_router(
    neural_network.router, prefix="/neural-network", tags=["Red Neuronal"]
)


@app.get("/", tags=["root"])
def root():
    return {"message": "Bienvenido a la API de XGBoost con FastAPI"}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
