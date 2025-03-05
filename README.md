# XGBoostBackend

Esta es una implementación de un backend para el modelo de XGBoost. Este backend se encarga de cargar el modelo y realizar predicciones en base a los datos de entrada.

### Instalación

1. Abrir el proyecto en un entorno de desarrollo o en su defecto, en una terminal.

2. Instalar las dependencias necesarias para el proyecto    , las cuales están en el archivo `requirements.txt`.

3. Para ejecutar el servicio, ejecutar el siguiente comando dentro de la raiz del proyecto:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

4. Probar conectividad abriendo en el navegador el siguiente enlace `http://localhost:8080/`. Debería aparecer un mensaje de bienvenida.

Con esto, el backend estará listo para recibir peticiones y realizar predicciones. Para usar este servicio, usar la ruta anterior como base en el frontend.