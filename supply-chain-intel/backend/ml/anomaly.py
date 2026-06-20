"""
Detects systemic warranty spikes using z-score on rolling 7-day claim counts per part.
Returns list of anomalies suitable for alerts.
"""
from datetime import date, timedelta
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session
from backend import models


def detect_warranty_anomalies(db: Session, lookback_days: int = 60, z_threshold: float = 2.0) -> list[dict]:
    cutoff = date.today() - timedelta(days=lookback_days)
    claims = db.query(models.WarrantyClaim).filter(
        models.WarrantyClaim.claim_date >= cutoff
    ).all()

    by_part: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    for c in claims:
        by_part[c.part_number][c.claim_date] += 1

    anomalies = []
    today = date.today()
    recent_window = today - timedelta(days=7)

    for part_number, daily_counts in by_part.items():
        history = [v for d, v in daily_counts.items() if d < recent_window]
        recent = [v for d, v in daily_counts.items() if d >= recent_window]
        if len(history) < 5 or not recent:
            continue
        mu = np.mean(history)
        sigma = np.std(history) or 1
        recent_avg = np.mean(recent)
        z = (recent_avg - mu) / sigma
        if z >= z_threshold:
            part_desc = next((c.part_description for c in claims if c.part_number == part_number), part_number)
            anomalies.append({
                "part_number": part_number,
                "part_description": part_desc,
                "z_score": round(float(z), 2),
                "recent_avg_claims_per_day": round(float(recent_avg), 2),
                "baseline_avg_claims_per_day": round(float(mu), 2),
                "severity": "critical" if z >= 3.0 else "warning",
            })

    anomalies.sort(key=lambda x: x["z_score"], reverse=True)
    return anomalies
