"""
Generates realistic synthetic data mimicking PACCAR's supply chain:
- Kenworth/Peterbilt parts catalog structure
- US distribution centers (Renton WA, Denton TX, Atlanta GA, etc.)
- Real carrier names, supplier profiles
- Realistic variance/noise on all metrics
"""
import random
from datetime import datetime, date, timedelta
from backend.connectors.base import BaseConnector

random.seed(42)

PARTS = [
    ("E-21401", "Engine Block Assembly 15L", "Powertrain"),
    ("E-21402", "Cylinder Head Assembly", "Powertrain"),
    ("T-44201", "18-Speed Transmission", "Drivetrain"),
    ("T-44202", "Clutch Pack Assembly", "Drivetrain"),
    ("A-88001", "Front Axle Assembly 12K", "Axle"),
    ("A-88002", "Rear Drive Axle 40K", "Axle"),
    ("B-33101", "Air Disc Brake Caliper LH", "Brakes"),
    ("B-33102", "Air Disc Brake Caliper RH", "Brakes"),
    ("S-77401", "Air Suspension Bag Front", "Suspension"),
    ("S-77402", "Shock Absorber Heavy Duty", "Suspension"),
    ("C-55001", "Cab Assembly Kenworth T680", "Cab"),
    ("C-55002", "Door Panel Assembly LH", "Cab"),
    ("F-11001", "DEF Tank 15 Gallon", "Emissions"),
    ("F-11002", "DPF Filter Assembly", "Emissions"),
    ("E-21403", "Turbocharger Assembly MX13", "Powertrain"),
    ("H-99001", "Hydraulic Pump Assembly", "Hydraulics"),
    ("W-66001", "Wiring Harness Main", "Electrical"),
    ("W-66002", "ECM Module PACCAR MX", "Electrical"),
    ("R-22001", "Radiator Heavy Duty 52in", "Cooling"),
    ("R-22002", "Water Pump Assembly", "Cooling"),
    ("G-44001", "Fifth Wheel 90000lb", "Coupling"),
    ("L-88001", "LED Headlamp Assembly LH", "Lighting"),
    ("P-33001", "Air Compressor 18CFM", "Air System"),
    ("P-33002", "Air Dryer Cartridge", "Air System"),
    ("N-11001", "Fuel Filter Primary", "Fuel System"),
]

DCS = ["Renton-WA", "Denton-TX", "Atlanta-GA", "Louisville-KY", "Toronto-ON"]

PLANTS = [
    ("KW-RENTON", "Kenworth Renton WA"),
    ("KW-CHILLICOTHE", "Kenworth Chillicothe OH"),
    ("PB-DENTON", "Peterbilt Denton TX"),
    ("PB-MADISON", "Peterbilt Madison TN"),
]

TRUCK_MODELS = ["T680", "T880", "W990", "379X", "389", "567", "579"]

SUPPLIERS = [
    ("SUP-001", "PACCAR Engines", "internal"),
    ("SUP-002", "Eaton Cummins JV", "USA"),
    ("SUP-003", "Dana Inc", "USA"),
    ("SUP-004", "Knorr-Bremse", "Germany"),
    ("SUP-005", "Hendrickson", "USA"),
    ("SUP-006", "Meritor", "USA"),
    ("SUP-007", "Wabco Holdings", "Belgium"),
    ("SUP-008", "Parker Hannifin", "USA"),
    ("SUP-009", "Belden Mfg", "USA"),
    ("SUP-010", "Modine Mfg", "USA"),
]

CARRIERS = ["JB Hunt", "Werner", "Schneider", "FedEx Freight", "XPO Logistics", "Estes Express", "Old Dominion"]

DEALERS = [
    ("DLR-001", "Pacific Kenworth Seattle", "Northwest"),
    ("DLR-002", "MHC Kenworth Dallas", "South Central"),
    ("DLR-003", "Peterbilt of Atlanta", "Southeast"),
    ("DLR-004", "Rush Truck Centers Houston", "South Central"),
    ("DLR-005", "Nussbaum Peterbilt Springfield", "Midwest"),
    ("DLR-006", "Thompson Truck & Trailer", "Northeast"),
    ("DLR-007", "Western Peterbilt Portland", "Northwest"),
    ("DLR-008", "Nextran Truck Centers Tampa", "Southeast"),
]

FAILURE_MODES = [
    "Oil Leak - Gasket Failure",
    "Premature Wear",
    "Electronic Malfunction",
    "Structural Crack",
    "Seal Failure",
    "Corrosion",
    "Bearing Failure",
    "Vibration Damage",
]


def _rand_date_range(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


class MockConnector(BaseConnector):
    name = "mock"

    def pull_inventory(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        for dc in DCS:
            for part_num, desc, family in PARTS:
                on_hand = random.uniform(5, 500)
                on_order = random.uniform(0, 200)
                reorder_pt = random.uniform(20, 100)
                records.append({
                    "snapshot_date": today,
                    "part_number": part_num,
                    "part_description": desc,
                    "part_family": family,
                    "location": dc,
                    "location_type": "dc",
                    "qty_on_hand": round(on_hand, 1),
                    "qty_on_order": round(on_order, 1),
                    "qty_reserved": round(on_hand * random.uniform(0.1, 0.4), 1),
                    "qty_available": round(max(on_hand - on_hand * 0.2, 0), 1),
                    "unit_cost": round(random.uniform(50, 8500), 2),
                    "reorder_point": round(reorder_pt, 1),
                    "source": "mock_sap",
                })
        return records

    def pull_supplier_orders(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        for i in range(150):
            sup = random.choice(SUPPLIERS)
            part = random.choice(PARTS)
            po_date = _rand_date_range(today - timedelta(days=90), today)
            promised = po_date + timedelta(days=random.randint(7, 45))
            late = random.random() < 0.12  # 12% late rate
            actual = promised + timedelta(days=random.randint(1, 10)) if late and promised <= today else (
                promised - timedelta(days=random.randint(0, 2)) if promised <= today else None
            )
            records.append({
                "po_number": f"PO-{10000 + i}",
                "supplier_id": sup[0],
                "supplier_name": sup[1],
                "part_number": part[0],
                "qty_ordered": random.randint(10, 500),
                "qty_received": random.randint(0, 500) if actual else 0,
                "unit_cost": round(random.uniform(50, 8500), 2),
                "po_date": po_date,
                "promised_date": promised,
                "actual_date": actual,
                "status": "complete" if actual else ("open" if promised > today else "overdue"),
                "plant": random.choice(PLANTS)[0],
                "is_critical": random.random() < 0.08,
                "source": "mock_sap",
            })
        return records

    def pull_production_output(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        for plant_id, plant_name in PLANTS:
            for offset in range(30):
                d = today - timedelta(days=offset)
                if d.weekday() >= 5:
                    continue
                planned = random.randint(18, 32)
                attainment = random.uniform(0.88, 1.02)
                built = int(planned * attainment)
                downtime = random.randint(0, 120) if attainment < 0.95 else random.randint(0, 30)
                records.append({
                    "output_date": d,
                    "plant": plant_id,
                    "plant_name": plant_name,
                    "model_family": random.choice(TRUCK_MODELS),
                    "units_planned": planned,
                    "units_built": built,
                    "downtime_minutes": downtime,
                    "downtime_reason": random.choice(["Parts shortage", "Equipment", "Quality hold", None]) if downtime > 60 else None,
                    "attainment_pct": round(built / planned * 100, 1),
                    "source": "mock_mes",
                })
        return records

    def pull_dealer_inventory(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        for dealer_id, dealer_name, region in DEALERS:
            for part_num, desc, family in random.sample(PARTS, 15):
                qty = random.uniform(0, 80)
                avg_daily = random.uniform(0.5, 5)
                doh = qty / avg_daily if avg_daily > 0 else 999
                records.append({
                    "snapshot_date": today,
                    "dealer_id": dealer_id,
                    "dealer_name": dealer_name,
                    "region": region,
                    "part_number": part_num,
                    "qty_on_hand": round(qty, 1),
                    "days_on_hand": round(doh, 1),
                    "last_movement_date": _rand_date_range(today - timedelta(days=60), today),
                    "avg_daily_demand": round(avg_daily, 2),
                    "source": "mock_cdk",
                })
        return records

    def pull_freight_shipments(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        for i in range(80):
            carrier = random.choice(CARRIERS)
            mode = random.choices(["ground", "ltl", "ftl", "air"], weights=[0.4, 0.35, 0.15, 0.1])[0]
            ship_date = _rand_date_range(today - timedelta(days=30), today)
            transit = {"ground": 3, "ltl": 5, "ftl": 4, "air": 1}[mode]
            promised = ship_date + timedelta(days=transit)
            late = random.random() < 0.08
            days_late = random.randint(1, 7) if late else 0
            actual = promised + timedelta(days=days_late) if promised <= today else None
            records.append({
                "shipment_id": f"SHP-{20000 + i}",
                "carrier": carrier,
                "origin": random.choice(DCS),
                "destination": random.choice(DEALERS)[0],
                "mode": mode,
                "ship_date": ship_date,
                "promised_eta": promised,
                "actual_eta": actual,
                "is_late": late,
                "days_late": days_late,
                "freight_cost": round(random.uniform(150, 4500), 2),
                "weight_lbs": round(random.uniform(50, 2000), 1),
                "status": "delivered" if actual else "in_transit",
                "exception_reason": "Carrier delay" if late else None,
                "source": "mock_tms",
            })
        return records

    def pull_warranty_claims(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        # Inject a systemic spike on E-21403 (turbocharger) to test anomaly detection
        spike_part = "E-21403"
        for i in range(60):
            part = random.choice(PARTS)
            if random.random() < 0.25:
                part = next(p for p in PARTS if p[0] == spike_part)
            plant = random.choice(PLANTS)
            sup = random.choice(SUPPLIERS)
            claim_date = _rand_date_range(today - timedelta(days=60), today)
            records.append({
                "claim_id": f"WC-{30000 + i}",
                "claim_date": claim_date,
                "part_number": part[0],
                "part_description": part[1],
                "truck_vin": f"1XKWDB9X{random.randint(10000000, 99999999)}9",
                "build_date": claim_date - timedelta(days=random.randint(90, 730)),
                "plant": plant[0],
                "supplier_id": sup[0],
                "failure_mode": random.choice(FAILURE_MODES),
                "repair_cost": round(random.uniform(200, 12000), 2),
                "is_systemic": part[0] == spike_part,
                "source": "mock_warranty",
            })
        return records

    def pull_demand_actuals(self, since: datetime) -> list[dict]:
        today = date.today()
        records = []
        regions = ["Northwest", "South Central", "Southeast", "Midwest", "Northeast"]
        for part_num, _, _ in PARTS:
            for region in regions:
                for offset in range(30):
                    d = today - timedelta(days=offset)
                    base_demand = random.uniform(2, 20)
                    # Weekend dip
                    if d.weekday() >= 5:
                        base_demand *= 0.3
                    qty = max(0, round(base_demand + random.gauss(0, 2), 1))
                    records.append({
                        "demand_date": d,
                        "part_number": part_num,
                        "region": region,
                        "qty_sold": qty,
                        "revenue": round(qty * random.uniform(50, 8500), 2),
                        "channel": random.choices(["dealer", "direct", "fleet"], weights=[0.6, 0.25, 0.15])[0],
                        "source": "mock_sap",
                    })
        return records
