@echo off

echo Stopping FastAPI...
taskkill /IM uvicorn.exe /F >nul 2>&1

echo Stopping Streamlit...
taskkill /IM streamlit.exe /F >nul 2>&1

echo All services stopped.
pause
