@echo off
title ArchiveGov AI — Government Records Restorer
echo =====================================================================
echo ArchiveGov AI — Government Records Restorer
echo Reviving Government History with Generative AI
echo =====================================================================
echo.

echo [1/2] Seeding archival demonstration records and security roles...
py scripts/seed_database.py
if %ERRORLEVEL% NEQ 0 (
    echo Seeding encountered a warning or completed with fallback. Proceeding...
)

echo.
echo [2/2] Starting Sovereign Uvicorn Application Server...
echo Portal URL: http://127.0.0.1:8000
echo Login URL:  http://127.0.0.1:8000/login
echo Swagger API Docs: http://127.0.0.1:8000/docs
echo.
echo Launching default browser to http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo.

py -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
pause

