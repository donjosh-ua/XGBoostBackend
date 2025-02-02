import uvicorn
from fastapi import FastAPI
from app.routes import train, predict

app = FastAPI(
    title="XGBoost with FastAPI",
    description="API para entrenar y predecir con modelos XGBoost y ajuste bayesiano.",
    version="1.0.0"
)

# Incluir routers
app.include_router(train.router, prefix="/train", tags=["Entrenamiento"])
app.include_router(predict.router, prefix="/predict", tags=["Predicción"])

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de XGBoost con FastAPI"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.0", port=8000)