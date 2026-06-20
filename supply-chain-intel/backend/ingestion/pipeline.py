import time
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.connectors import get_connectors
from backend import models


def _upsert_inventory(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.InventorySnapshot).filter_by(
            snapshot_date=r["snapshot_date"],
            part_number=r["part_number"],
            location=r["location"],
        ).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.InventorySnapshot(**r))


def _upsert_supplier_orders(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.SupplierOrder).filter_by(po_number=r["po_number"]).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.SupplierOrder(**r))


def _upsert_production(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.ProductionOutput).filter_by(
            output_date=r["output_date"], plant=r["plant"]
        ).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.ProductionOutput(**r))


def _upsert_dealer_inventory(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.DealerInventory).filter_by(
            snapshot_date=r["snapshot_date"],
            dealer_id=r["dealer_id"],
            part_number=r["part_number"],
        ).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.DealerInventory(**r))


def _upsert_freight(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.FreightShipment).filter_by(shipment_id=r["shipment_id"]).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.FreightShipment(**r))


def _upsert_warranty(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.WarrantyClaim).filter_by(claim_id=r["claim_id"]).first()
        if not existing:
            db.add(models.WarrantyClaim(**r))


def _upsert_demand(db: Session, records: list[dict]):
    for r in records:
        existing = db.query(models.DemandActual).filter_by(
            demand_date=r["demand_date"],
            part_number=r["part_number"],
            region=r["region"],
        ).first()
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            db.add(models.DemandActual(**r))


def run_ingestion(db: Session, since: datetime | None = None) -> dict:
    if since is None:
        since = datetime.utcnow() - timedelta(hours=24)

    results = {}
    for connector in get_connectors():
        t0 = time.time()
        log = models.ConnectorLog(connector_name=connector.name)
        try:
            inv = connector.pull_inventory(since)
            _upsert_inventory(db, inv)

            pos = connector.pull_supplier_orders(since)
            _upsert_supplier_orders(db, pos)

            prod = connector.pull_production_output(since)
            _upsert_production(db, prod)

            dlr = connector.pull_dealer_inventory(since)
            _upsert_dealer_inventory(db, dlr)

            freight = connector.pull_freight_shipments(since)
            _upsert_freight(db, freight)

            warranty = connector.pull_warranty_claims(since)
            _upsert_warranty(db, warranty)

            demand = connector.pull_demand_actuals(since)
            _upsert_demand(db, demand)

            db.commit()
            total = len(inv) + len(pos) + len(prod) + len(dlr) + len(freight) + len(warranty) + len(demand)
            log.status = "success"
            log.records_pulled = total
            results[connector.name] = {"status": "success", "records": total}
        except Exception as e:
            db.rollback()
            log.status = "error"
            log.error_message = str(e)
            results[connector.name] = {"status": "error", "error": str(e)}
        finally:
            log.duration_ms = int((time.time() - t0) * 1000)
            db.add(log)
            db.commit()

    return results
