@echo off
call "%~dp0venv\Scripts\activate.bat"
REM Use the SAME backend as training (SQLite file in your project root)
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

