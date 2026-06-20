import json
from datetime import date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from backend import models


def compute_weekly_metrics(db: Session, week_start: date | None = None) -> models.WeeklyMetric:
    if week_start is None:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)

    # Inventory turns: (COGS in period / avg inventory value) * 52
    inv = db.query(models.InventorySnapshot).filter(
        models.InventorySnapshot.snapshot_date >= week_start,
        models.InventorySnapshot.snapshot_date <= week_end,
    ).all()
    avg_inv_value = sum((i.qty_on_hand or 0) * (i.unit_cost or 0) for i in inv) / max(len(set(i.snapshot_date for i in inv)), 1)

    demand = db.query(models.DemandActual).filter(
        models.DemandActual.demand_date >= week_start,
        models.DemandActual.demand_date <= week_end,
    ).all()
    weekly_cogs = sum(d.revenue or 0 for d in demand) * 0.65  # rough COGS at 65% of revenue
    inv_turns = (weekly_cogs / avg_inv_value * 52) if avg_inv_value > 0 else 0

    # Forecast accuracy (MAPE) — compare last week demand vs simple moving avg
    prev_week_start = week_start - timedelta(days=7)
    prev_demand = db.query(models.DemandActual).filter(
        models.DemandActual.demand_date >= prev_week_start,
        models.DemandActual.demand_date < week_start,
    ).all()
    forecast_mape = _compute_mape(demand, prev_demand)

    # Avg daily metrics for the week
    daily_metrics = db.query(models.DailyMetric).filter(
        models.DailyMetric.metric_date >= week_start,
        models.DailyMetric.metric_date <= week_end,
    ).all()

    avg_fill_rate = np.mean([m.fill_rate_pct for m in daily_metrics if m.fill_rate_pct]) if daily_metrics else 0
    avg_otd = np.mean([m.supplier_otd_pct for m in daily_metrics if m.supplier_otd_pct]) if daily_metrics else 0
    avg_attainment = np.mean([m.production_attainment_pct for m in daily_metrics if m.production_attainment_pct]) if daily_metrics else 0

    # Freight cost for week
    freight = db.query(models.FreightShipment).filter(
        models.FreightShipment.ship_date >= week_start,
        models.FreightShipment.ship_date <= week_end,
    ).all()
    total_freight_cost = sum(f.freight_cost or 0 for f in freight)

    # Warranty cost for week
    warranty = db.query(models.WarrantyClaim).filter(
        models.WarrantyClaim.claim_date >= week_start,
        models.WarrantyClaim.claim_date <= week_end,
    ).all()
    total_warranty_cost = sum(w.repair_cost or 0 for w in warranty)

    # Top warranty part
    if warranty:
        from collections import Counter
        part_counts = Counter(w.part_number for w in warranty)
        top_warranty_part = part_counts.most_common(1)[0][0]
    else:
        top_warranty_part = None

    # PO value for week
    pos = db.query(models.SupplierOrder).filter(
        models.SupplierOrder.po_date >= week_start,
        models.SupplierOrder.po_date <= week_end,
    ).all()
    total_po_value = sum((p.qty_ordered or 0) * (p.unit_cost or 0) for p in pos)

    # Excess and short inventory
    latest_snap = db.query(func.max(models.InventorySnapshot.snapshot_date)).scalar()
    current_inv = db.query(models.InventorySnapshot).filter_by(snapshot_date=latest_snap).all() if latest_snap else []
    excess_value = sum((i.qty_on_hand or 0) * (i.unit_cost or 0) for i in current_inv
                      if i.qty_on_hand and i.reorder_point and i.qty_on_hand > i.reorder_point * 3)
    short_parts = sum(1 for i in current_inv
                     if i.qty_available is not None and i.reorder_point and i.qty_available < i.reorder_point)

    raw = {
        "week_start": str(week_start),
        "week_end": str(week_end),
        "inventory_turns": round(float(inv_turns), 2),
        "forecast_mape": round(float(forecast_mape), 2),
        "avg_fill_rate_pct": round(float(avg_fill_rate), 2),
        "avg_supplier_otd_pct": round(float(avg_otd), 2),
        "avg_production_attainment_pct": round(float(avg_attainment), 2),
        "total_freight_cost": round(total_freight_cost, 2),
        "total_warranty_cost": round(total_warranty_cost, 2),
        "total_po_value": round(total_po_value, 2),
        "excess_inventory_value": round(excess_value, 2),
        "short_inventory_parts": short_parts,
        "top_warranty_part": top_warranty_part,
    }

    existing = db.query(models.WeeklyMetric).filter_by(week_start=week_start).first()
    if existing:
        for k, v in raw.items():
            if k in ("week_start", "week_end"):
                continue
            if hasattr(existing, k):
                setattr(existing, k, v)
        existing.week_end = week_end
        existing.raw_json = json.dumps(raw)
        db.commit()
        return existing

    db_fields = {k: v for k, v in raw.items() if k not in ("week_start", "week_end")}
    metric = models.WeeklyMetric(week_start=week_start, week_end=week_end, raw_json=json.dumps(raw), **db_fields)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def _compute_mape(actuals: list, forecasts: list) -> float:
    if not actuals or not forecasts:
        return 0.0
    act_by_part = {}
    for d in actuals:
        act_by_part.setdefault(d.part_number, []).append(d.qty_sold or 0)
    fc_by_part = {}
    for d in forecasts:
        fc_by_part.setdefault(d.part_number, []).append(d.qty_sold or 0)

    errors = []
    for part, act_vals in act_by_part.items():
        if part in fc_by_part:
            act_mean = np.mean(act_vals)
            fc_mean = np.mean(fc_by_part[part])
            if act_mean > 0:
                errors.append(abs(act_mean - fc_mean) / act_mean * 100)
    return float(np.mean(errors)) if errors else 0.0
