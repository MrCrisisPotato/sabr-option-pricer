@echo off
SETLOCAL

REM 
SET CONDA_PATH=C:\Users\Sourav\anaconda3

echo Activating conda environment...
CALL "%CONDA_PATH%\Scripts\activate.bat" sabr_cuda

cd backend
echo Starting FastAPI backend...
start cmd /k uvicorn main:app --reload

cd ..
echo Starting Streamlit frontend...
start cmd /k streamlit run streamlit_app.py

ENDLOCAL
