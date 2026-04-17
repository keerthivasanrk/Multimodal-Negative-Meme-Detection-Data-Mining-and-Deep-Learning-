@echo off
echo Starting Multimodal Meme Moderation System...

:: Start Backend in a new window .\start_everything.bat
echo Starting AI Backend (Port 8000)...
start "Backend - FastAPI" cmd /k ".\venv310\Scripts\activate && python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload"

:: Start Frontend in a new window
echo Starting React Frontend (Port 5173)...
start "Frontend - Vite" cmd /k "cd frontend && npm run dev"

echo.
echo All services are launching! 
echo Once "Application startup complete" appears in the Backend window,
echo open http://localhost:5173 in your browser.
echo.
pause
