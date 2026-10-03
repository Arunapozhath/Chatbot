@echo off
echo ============================================================
echo  Milkosoft AI Chatbot v2
echo ============================================================

if not exist venv (
    echo ERROR: venv not found. Run setup.bat first.
    pause & exit /b 1
)

call venv\Scripts\activate.bat

if not exist chroma_db (
    echo WARNING: ChromaDB not found.
    echo          Run: python ingest.py --dir data
    echo          to index your knowledge base documents first.
    echo.
)

echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python app.py
pause
