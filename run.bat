@echo off
setlocal
cd /d "%~dp0"
if not exist "venv311\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run install.bat first.
  pause
  exit /b 1
)
call venv311\Scripts\activate.bat
python app.py
pause
