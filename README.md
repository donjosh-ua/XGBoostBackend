# XGBoostBackend

Esta es una implementación de un backend para el modelo de XGBoost. Este backend se encarga de cargar el modelo y realizar predicciones en base a los datos de entrada.

### Instalación

1. Abrir este proyecto en una terminal. Para esto se debe abrir la carpeta del proyecto. Dentro del proyecto, dar click derecho y seleccionar la opción **"Abrir en terminal"**. Una alternativa a esto es abrir una terminal y navegar hasta la carpeta del proyecto mediante el comando `cd`.

2. Una vez en la terminal, ejecutar el siguiente comando para instalar las dependencias del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Para ejecutar el backend, ejecutar el siguiente comando:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
