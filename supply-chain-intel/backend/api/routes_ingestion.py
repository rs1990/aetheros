import csv
import io
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.ingestion.pipeline import run_ingestion
from backend import models

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


@router.post("/run")
def trigger_ingestion(db: Session = Depends(get_db)):
    results = run_ingestion(db)
    return {"status": "complete", "connectors": results}


@router.get("/logs")
def get_connector_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(models.ConnectorLog).order_by(
        models.ConnectorLog.run_at.desc()
    ).limit(limit).all()
    return [
        {
            "connector": l.connector_name,
            "status": l.status,
            "records": l.records_pulled,
            "error": l.error_message,
            "duration_ms": l.duration_ms,
            "run_at": l.run_at.isoformat() if l.run_at else None,
        }
        for l in logs
    ]


@router.post("/upload/inventory")
async def upload_inventory_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    required = {"part_number", "location", "qty_on_hand", "snapshot_date"}
    rows = list(reader)
    if not rows:
        raise HTTPException(400, "Empty CSV")
    missing = required - set(rows[0].keys())
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")
    count = 0
    for row in rows:
        record = models.InventorySnapshot(
            snapshot_date=row["snapshot_date"],
            part_number=row["part_number"],
            part_description=row.get("part_description"),
            part_family=row.get("part_family"),
            location=row["location"],
            location_type=row.get("location_type", "dc"),
            qty_on_hand=float(row["qty_on_hand"] or 0),
            qty_on_order=float(row.get("qty_on_order") or 0),
            qty_reserved=float(row.get("qty_reserved") or 0),
            qty_available=float(row.get("qty_available") or 0),
            unit_cost=float(row.get("unit_cost") or 0),
            reorder_point=float(row.get("reorder_point") or 0),
            source="csv_upload",
        )
        db.add(record)
        count += 1
    db.commit()
    return {"uploaded": count}
