import json
from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database import get_db
from backend import models
from backend.metrics.daily import compute_daily_metrics
from backend.metrics.weekly import compute_weekly_metrics
from backend.ml.anomaly import detect_warranty_anomalies
from backend.ml.forecast import forecast_demand

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/daily")
def get_daily_metrics(
    days: int = Query(30, ge=1, le=90),
    db: Session = Depends(get_db)
):
    records = db.query(models.DailyMetric).order_by(
        models.DailyMetric.metric_date.desc()
    ).limit(days).all()
    return [_daily_to_dict(r) for r in reversed(records)]


@router.get("/daily/today")
def get_today_metrics(db: Session = Depends(get_db)):
    metric = db.query(models.DailyMetric).filter_by(metric_date=date.today()).first()
    if not metric:
        metric = compute_daily_metrics(db)
    raw = json.loads(metric.raw_json) if metric.raw_json else {}
    return {**_daily_to_dict(metric), "alerts": raw.get("alerts", [])}


@router.post("/daily/compute")
def trigger_daily_compute(db: Session = Depends(get_db)):
    metric = compute_daily_metrics(db)
    raw = json.loads(metric.raw_json) if metric.raw_json else {}
    return {**_daily_to_dict(metric), "alerts": raw.get("alerts", [])}


@router.get("/weekly")
def get_weekly_metrics(
    weeks: int = Query(12, ge=1, le=52),
    db: Session = Depends(get_db)
):
    records = db.query(models.WeeklyMetric).order_by(
        models.WeeklyMetric.week_start.desc()
    ).limit(weeks).all()
    return [_weekly_to_dict(r) for r in reversed(records)]


@router.post("/weekly/compute")
def trigger_weekly_compute(db: Session = Depends(get_db)):
    metric = compute_weekly_metrics(db)
    return _weekly_to_dict(metric)


@router.get("/anomalies")
def get_warranty_anomalies(db: Session = Depends(get_db)):
    return detect_warranty_anomalies(db)


@router.get("/forecast")
def get_demand_forecast(db: Session = Depends(get_db)):
    return forecast_demand(db)


@router.get("/inventory/summary")
def get_inventory_summary(db: Session = Depends(get_db)):
    latest = db.query(func.max(models.InventorySnapshot.snapshot_date)).scalar()
    if not latest:
        return {"total_skus": 0, "in_stock": 0, "total_value": 0, "by_location": []}
    inv = db.query(models.InventorySnapshot).filter_by(snapshot_date=latest).all()
    by_loc = {}
    for i in inv:
        loc = i.location
        if loc not in by_loc:
            by_loc[loc] = {"location": loc, "skus": 0, "value": 0, "in_stock": 0}
        by_loc[loc]["skus"] += 1
        by_loc[loc]["value"] += (i.qty_on_hand or 0) * (i.unit_cost or 0)
        if i.qty_available and i.qty_available > 0:
            by_loc[loc]["in_stock"] += 1
    return {
        "snapshot_date": str(latest),
        "total_skus": len(inv),
        "in_stock": sum(1 for i in inv if i.qty_available and i.qty_available > 0),
        "total_value": round(sum((i.qty_on_hand or 0) * (i.unit_cost or 0) for i in inv), 2),
        "by_location": [{"location": k, **v} for k, v in by_loc.items()],
    }


@router.get("/suppliers/performance")
def get_supplier_performance(db: Session = Depends(get_db)):
    window = date.today() - timedelta(days=30)
    pos = db.query(models.SupplierOrder).filter(
        models.SupplierOrder.po_date >= window
    ).all()
    by_sup = {}
    for p in pos:
        sid = p.supplier_id
        if sid not in by_sup:
            by_sup[sid] = {"supplier_id": sid, "name": p.supplier_name, "total": 0, "on_time": 0, "late": 0, "open": 0}
        by_sup[sid]["total"] += 1
        if p.actual_date:
            if p.promised_date and p.actual_date <= p.promised_date:
                by_sup[sid]["on_time"] += 1
            else:
                by_sup[sid]["late"] += 1
        else:
            by_sup[sid]["open"] += 1
    result = []
    for s in by_sup.values():
        delivered = s["on_time"] + s["late"]
        s["otd_pct"] = round(s["on_time"] / delivered * 100, 1) if delivered > 0 else None
        result.append(s)
    return sorted(result, key=lambda x: (x["otd_pct"] or 100), reverse=False)


@router.get("/freight/summary")
def get_freight_summary(db: Session = Depends(get_db)):
    window = date.today() - timedelta(days=30)
    shipments = db.query(models.FreightShipment).filter(
        models.FreightShipment.ship_date >= window
    ).all()
    by_carrier = {}
    for s in shipments:
        c = s.carrier
        if c not in by_carrier:
            by_carrier[c] = {"carrier": c, "total": 0, "on_time": 0, "late": 0, "cost": 0}
        by_carrier[c]["total"] += 1
        by_carrier[c]["cost"] += s.freight_cost or 0
        if s.is_late:
            by_carrier[c]["late"] += 1
        elif s.actual_eta:
            by_carrier[c]["on_time"] += 1
    result = []
    for c in by_carrier.values():
        delivered = c["on_time"] + c["late"]
        c["otd_pct"] = round(c["on_time"] / delivered * 100, 1) if delivered > 0 else None
        c["cost"] = round(c["cost"], 2)
        result.append(c)
    return sorted(result, key=lambda x: x["total"], reverse=True)


@router.get("/warranty/summary")
def get_warranty_summary(db: Session = Depends(get_db)):
    window = date.today() - timedelta(days=30)
    claims = db.query(models.WarrantyClaim).filter(
        models.WarrantyClaim.claim_date >= window
    ).all()
    by_part = {}
    for c in claims:
        p = c.part_number
        if p not in by_part:
            by_part[p] = {"part_number": p, "description": c.part_description, "count": 0, "cost": 0}
        by_part[p]["count"] += 1
        by_part[p]["cost"] += c.repair_cost or 0
    result = sorted(by_part.values(), key=lambda x: x["count"], reverse=True)
    for r in result:
        r["cost"] = round(r["cost"], 2)
    return result[:15]


def _daily_to_dict(m: models.DailyMetric) -> dict:
    return {
        "date": str(m.metric_date),
        "fill_rate_pct": m.fill_rate_pct,
        "open_backorders": m.open_backorders,
        "critical_backorders": m.critical_backorders,
        "supplier_otd_pct": m.supplier_otd_pct,
        "production_attainment_pct": m.production_attainment_pct,
        "freight_exception_count": m.freight_exception_count,
        "freight_on_time_pct": m.freight_on_time_pct,
        "warranty_claims_count": m.warranty_claims_count,
        "warranty_cost": m.warranty_cost,
        "inventory_value": m.inventory_value,
        "active_pos": m.active_pos,
        "po_overdue_count": m.po_overdue_count,
    }


def _weekly_to_dict(m: models.WeeklyMetric) -> dict:
    return {
        "week_start": str(m.week_start),
        "week_end": str(m.week_end),
        "inventory_turns": m.inventory_turns,
        "forecast_mape": m.forecast_mape,
        "avg_fill_rate_pct": m.avg_fill_rate_pct,
        "avg_supplier_otd_pct": m.avg_supplier_otd_pct,
        "avg_production_attainment_pct": m.avg_production_attainment_pct,
        "total_freight_cost": m.total_freight_cost,
        "total_warranty_cost": m.total_warranty_cost,
        "total_po_value": m.total_po_value,
        "excess_inventory_value": m.excess_inventory_value,
        "short_inventory_parts": m.short_inventory_parts,
        "top_warranty_part": m.top_warranty_part,
    }
