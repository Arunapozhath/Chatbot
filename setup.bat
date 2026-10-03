@echo off
echo ============================================================
echo  Milkosoft AI Chatbot v2 - Setup
echo ============================================================

echo.
echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+
    pause & exit /b 1
)

echo.
echo [2/4] Activating venv and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt
    pause & exit /b 1
)

echo.
echo [3/4] Pulling Ollama models (requires Ollama to be running)...
echo   Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama not running. Start it first with: ollama serve
    echo          Then manually pull models:
    echo            ollama pull llama3.2
    echo            ollama pull nomic-embed-text
) else (
    echo   Pulling llama3.2...
    ollama pull llama3.2
    echo   Pulling nomic-embed-text...
    ollama pull nomic-embed-text
)

echo.
echo [4/4] Setup complete!
echo.
echo NEXT STEPS:
echo   1. Copy your knowledge base documents into the  data\  folder
echo      Supported: .txt .md .docx .pdf .xlsx .xls .csv
echo.
echo   2. Run ingestion:
echo      venv\Scripts\activate.bat
echo      python ingest.py
echo.
echo   3. Start the chatbot:
echo      start.bat
echo.
pause
