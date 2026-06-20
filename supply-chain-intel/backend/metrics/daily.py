import json
from datetime import date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend import models


def compute_daily_metrics(db: Session, for_date: date | None = None) -> models.DailyMetric:
    d = for_date or date.today()

    # Fill rate: parts with stock / total parts across DCs
    inv = db.query(models.InventorySnapshot).filter_by(snapshot_date=d).all()
    if not inv:
        # Fall back to most recent snapshot
        latest = db.query(func.max(models.InventorySnapshot.snapshot_date)).scalar()
        inv = db.query(models.InventorySnapshot).filter_by(snapshot_date=latest).all() if latest else []

    total_skus = len(inv)
    in_stock = sum(1 for i in inv if i.qty_available and i.qty_available > 0)
    fill_rate = (in_stock / total_skus * 100) if total_skus > 0 else 0

    # Backorders: POs where promised_date < today and not complete
    open_pos = db.query(models.SupplierOrder).filter(
        models.SupplierOrder.status.in_(["open", "overdue"])
    ).all()
    open_backorders = len(open_pos)
    critical_backorders = sum(1 for p in open_pos if p.is_critical)
    po_overdue = sum(1 for p in open_pos if p.promised_date and p.promised_date < d)

    # Supplier OTD: completed POs in last 30 days
    window = d - timedelta(days=30)
    completed = db.query(models.SupplierOrder).filter(
        models.SupplierOrder.actual_date.isnot(None),
        models.SupplierOrder.actual_date >= window,
        models.SupplierOrder.actual_date <= d,
    ).all()
    otd_count = sum(1 for p in completed if p.actual_date and p.promised_date and p.actual_date <= p.promised_date)
    supplier_otd = (otd_count / len(completed) * 100) if completed else 0

    # Production attainment: today or most recent weekday
    prod = db.query(models.ProductionOutput).filter_by(output_date=d).all()
    if not prod:
        latest_prod = db.query(func.max(models.ProductionOutput.output_date)).scalar()
        prod = db.query(models.ProductionOutput).filter_by(output_date=latest_prod).all() if latest_prod else []
    total_planned = sum(p.units_planned for p in prod) or 1
    total_built = sum(p.units_built for p in prod)
    production_attainment = total_built / total_planned * 100

    # Freight
    freight = db.query(models.FreightShipment).filter(
        models.FreightShipment.ship_date >= d - timedelta(days=7)
    ).all()
    exception_count = sum(1 for f in freight if f.is_late)
    delivered = [f for f in freight if f.actual_eta is not None]
    on_time_freight = sum(1 for f in delivered if not f.is_late)
    freight_on_time_pct = (on_time_freight / len(delivered) * 100) if delivered else 100

    # Warranty
    warranty = db.query(models.WarrantyClaim).filter(
        models.WarrantyClaim.claim_date >= d - timedelta(days=7),
        models.WarrantyClaim.claim_date <= d,
    ).all()
    warranty_count = len(warranty)
    warranty_cost = sum(w.repair_cost or 0 for w in warranty)

    # Inventory value
    inv_value = sum((i.qty_on_hand or 0) * (i.unit_cost or 0) for i in inv)

    raw = {
        "fill_rate_pct": round(fill_rate, 2),
        "open_backorders": open_backorders,
        "critical_backorders": critical_backorders,
        "supplier_otd_pct": round(supplier_otd, 2),
        "production_attainment_pct": round(production_attainment, 2),
        "freight_exception_count": exception_count,
        "freight_on_time_pct": round(freight_on_time_pct, 2),
        "warranty_claims_count": warranty_count,
        "warranty_cost": round(warranty_cost, 2),
        "inventory_value": round(inv_value, 2),
        "active_pos": len(open_pos),
        "po_overdue_count": po_overdue,
        "alerts": _build_alerts(fill_rate, critical_backorders, supplier_otd, production_attainment, exception_count),
    }

    existing = db.query(models.DailyMetric).filter_by(metric_date=d).first()
    if existing:
        for k, v in raw.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        existing.raw_json = json.dumps(raw)
        db.commit()
        return existing

    metric = models.DailyMetric(metric_date=d, raw_json=json.dumps(raw), **{k: v for k, v in raw.items() if k != "alerts"})
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def _build_alerts(fill_rate, critical_backorders, supplier_otd, production_attainment, freight_exceptions) -> list[dict]:
    alerts = []
    if fill_rate < 95:
        alerts.append({"level": "warning", "metric": "fill_rate", "message": f"Fill rate {fill_rate:.1f}% below 95% target"})
    if critical_backorders > 0:
        alerts.append({"level": "critical", "metric": "backorders", "message": f"{critical_backorders} critical parts on backorder"})
    if supplier_otd < 90:
        alerts.append({"level": "warning", "metric": "supplier_otd", "message": f"Supplier OTD {supplier_otd:.1f}% below 90% target"})
    if production_attainment < 95:
        alerts.append({"level": "warning", "metric": "production", "message": f"Production attainment {production_attainment:.1f}% below 95% target"})
    if freight_exceptions > 5:
        alerts.append({"level": "warning", "metric": "freight", "message": f"{freight_exceptions} freight exceptions in last 7 days"})
    return alerts
