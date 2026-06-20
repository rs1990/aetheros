from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from backend.database import SessionLocal
from backend.ingestion.pipeline import run_ingestion
from backend.metrics.daily import compute_daily_metrics
from backend.metrics.weekly import compute_weekly_metrics

scheduler = BackgroundScheduler()


def _pull_data():
    db = SessionLocal()
    try:
        run_ingestion(db)
    finally:
        db.close()


def _daily_metrics():
    db = SessionLocal()
    try:
        compute_daily_metrics(db)
    finally:
        db.close()


def _weekly_metrics():
    db = SessionLocal()
    try:
        compute_weekly_metrics(db)
    finally:
        db.close()


def start():
    # Pull data every 15 minutes
    scheduler.add_job(_pull_data, "interval", minutes=15, id="pull_data")
    # Compute daily metrics at 6am
    scheduler.add_job(_daily_metrics, CronTrigger(hour=6, minute=0), id="daily_metrics")
    # Compute weekly metrics Monday 7am
    scheduler.add_job(_weekly_metrics, CronTrigger(day_of_week="mon", hour=7, minute=0), id="weekly_metrics")
    scheduler.start()


def stop():
    scheduler.shutdown()
