from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.database import init_db
from backend.ingestion.pipeline import run_ingestion
from backend.database import SessionLocal
from backend.metrics.daily import compute_daily_metrics
from backend.metrics.weekly import compute_weekly_metrics
from backend.scheduler import start as start_scheduler, stop as stop_scheduler
from backend.api.routes_metrics import router as metrics_router
from backend.api.routes_ingestion import router as ingestion_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        run_ingestion(db)
        compute_daily_metrics(db)
        compute_weekly_metrics(db)
    finally:
        db.close()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Supply Chain Intelligence", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router)
app.include_router(ingestion_router)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
