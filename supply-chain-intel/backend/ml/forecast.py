"""
Simple demand forecasting using linear trend + 7-day moving average.
Returns 14-day forward forecast per part per region.
"""
from datetime import date, timedelta
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session
from backend import models


def forecast_demand(db: Session, horizon_days: int = 14) -> list[dict]:
    cutoff = date.today() - timedelta(days=60)
    actuals = db.query(models.DemandActual).filter(
        models.DemandActual.demand_date >= cutoff
    ).all()

    by_part_region: dict[tuple, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    for a in actuals:
        by_part_region[(a.part_number, a.region)][a.demand_date] += (a.qty_sold or 0)

    forecasts = []
    today = date.today()

    for (part_number, region), daily_qty in by_part_region.items():
        sorted_dates = sorted(daily_qty.keys())
        if len(sorted_dates) < 7:
            continue
        values = [daily_qty[d] for d in sorted_dates]
        x = np.arange(len(values))
        # Fit linear trend
        slope, intercept = np.polyfit(x, values, 1)
        ma7 = np.mean(values[-7:])

        for i in range(1, horizon_days + 1):
            fcast_date = today + timedelta(days=i)
            trend_val = intercept + slope * (len(values) + i)
            # Blend trend and moving average
            predicted = max(0, 0.4 * trend_val + 0.6 * ma7)
            # Weekend factor
            if fcast_date.weekday() >= 5:
                predicted *= 0.3
            forecasts.append({
                "part_number": part_number,
                "region": region,
                "forecast_date": fcast_date,
                "predicted_qty": round(float(predicted), 2),
                "ma7": round(float(ma7), 2),
            })

    return forecasts
