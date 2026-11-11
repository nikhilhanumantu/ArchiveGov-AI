import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager

from backend.app.core.config import settings, FRONTEND_DIR
from backend.app.database.mongodb import init_db
from backend.app.routers import (
    auth, documents, restoration, damage_analysis,
    inpainting, ocr, archive, versions, reviews,
    analytics, users, settings as settings_router,
    notifications, files, pages
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("archivegov")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MongoDB and check collections
    logger.info("Initializing ArchiveGov AI sovereign platform...")
    db = init_db()
    
    # Auto-seed if users table is empty
    from scripts.seed_database import seed_all
    try:
        seed_all(db)
    except Exception as e:
        logger.warning(f"Database seed note: {e}")
        
    yield
    logger.info("Shutting down ArchiveGov AI.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="ArchiveGov AI — Government Records Restorer with Generative AI",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
static_path = FRONTEND_DIR / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Mount API Routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(restoration.router)
app.include_router(damage_analysis.router)
app.include_router(inpainting.router)
app.include_router(ocr.router)
app.include_router(archive.router)
app.include_router(versions.router)
app.include_router(reviews.router)
app.include_router(analytics.router)
app.include_router(users.router)
app.include_router(settings_router.router)
app.include_router(notifications.router)
app.include_router(files.router)

# Mount HTML UI Pages Router
app.include_router(pages.router)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Return JSON for API routes, friendly redirect/message for web
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if exc.status_code == 401:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/login?redirect={request.url.path}")
    return HTMLResponse(
        content=f"""
        <div style="background:#070B1C;color:#dee1fa;font-family:sans-serif;height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column;">
            <h1 style="color:#efbf66;">ArchiveGov AI Error {exc.status_code}</h1>
            <p>{exc.detail}</p>
            <a href="/app/dashboard" style="color:#5de6ff;margin-top:20px;text-decoration:none;">&larr; Return to Dashboard</a>
        </div>
        """,
        status_code=exc.status_code
    )

