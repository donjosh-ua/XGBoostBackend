import uvicorn
from fastapi import FastAPI
from routes import training
from fastapi.middleware.cors import CORSMiddleware
from routes import predict, data_file, tunning, testing

app = FastAPI(
    title="XGBoost with FastAPI",
    description="API para entrenar y predecir con modelos XGBoost y ajuste bayesiano.",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:5173",    # frontend url
    "http://127.0.0.0:8000"     
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

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de XGBoost con FastAPI"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.0", port=8000)