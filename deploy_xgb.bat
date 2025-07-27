@echo off
REM Restore all files to the latest commit
git restore .
git restore --staged .
REM Pull latest changes from remote
git pull
REM Build Docker image with name xgb
docker build -t xgb .
REM Stop and remove any existing container named xgb
docker stop xgb 2>NUL
REM Remove the container if it exists
docker rm xgb 2>NUL
REM Run the Docker image, mapping port 8080
docker run -d --name xgb -p 8080:8080 xgb
