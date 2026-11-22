@echo off
setlocal
cd /d "%~dp0"
echo Creating Python 3.11 virtual environment...
py -3.11 -m venv venv311
call venv311\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check the error above.
  pause
  exit /b 1
)
echo.
echo Installation completed successfully.
pause
