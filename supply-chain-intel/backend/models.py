from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Date
from backend.database import Base


class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    part_number = Column(String(50), nullable=False, index=True)
    part_description = Column(String(200))
    part_family = Column(String(50))
    location = Column(String(50), nullable=False)   # DC or plant
    location_type = Column(String(20))              # dc, plant, dealer
    qty_on_hand = Column(Float, default=0)
    qty_on_order = Column(Float, default=0)
    qty_reserved = Column(Float, default=0)
    qty_available = Column(Float, default=0)
    unit_cost = Column(Float, default=0)
    reorder_point = Column(Float, default=0)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class SupplierOrder(Base):
    __tablename__ = "supplier_orders"
    id = Column(Integer, primary_key=True)
    po_number = Column(String(50), unique=True, nullable=False)
    supplier_id = Column(String(50), nullable=False, index=True)
    supplier_name = Column(String(100))
    part_number = Column(String(50), nullable=False)
    qty_ordered = Column(Float)
    qty_received = Column(Float, default=0)
    unit_cost = Column(Float)
    po_date = Column(Date)
    promised_date = Column(Date)
    actual_date = Column(Date, nullable=True)
    status = Column(String(20))     # open, partial, complete, cancelled
    plant = Column(String(20))
    is_critical = Column(Boolean, default=False)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class ProductionOutput(Base):
    __tablename__ = "production_output"
    id = Column(Integer, primary_key=True)
    output_date = Column(Date, nullable=False, index=True)
    plant = Column(String(20), nullable=False)
    plant_name = Column(String(100))
    model_family = Column(String(50))
    units_planned = Column(Integer)
    units_built = Column(Integer)
    downtime_minutes = Column(Integer, default=0)
    downtime_reason = Column(String(100), nullable=True)
    attainment_pct = Column(Float)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class DealerInventory(Base):
    __tablename__ = "dealer_inventory"
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    dealer_id = Column(String(20), nullable=False)
    dealer_name = Column(String(100))
    region = Column(String(50))
    part_number = Column(String(50), nullable=False)
    qty_on_hand = Column(Float, default=0)
    days_on_hand = Column(Float)
    last_movement_date = Column(Date, nullable=True)
    avg_daily_demand = Column(Float, default=0)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class FreightShipment(Base):
    __tablename__ = "freight_shipments"
    id = Column(Integer, primary_key=True)
    shipment_id = Column(String(50), unique=True, nullable=False)
    carrier = Column(String(50))
    origin = Column(String(50))
    destination = Column(String(50))
    mode = Column(String(20))       # ground, air, ltl, ftl, ocean
    ship_date = Column(Date)
    promised_eta = Column(Date)
    actual_eta = Column(Date, nullable=True)
    is_late = Column(Boolean, default=False)
    days_late = Column(Integer, default=0)
    freight_cost = Column(Float)
    weight_lbs = Column(Float)
    status = Column(String(20))     # in_transit, delivered, exception, cancelled
    exception_reason = Column(String(100), nullable=True)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class WarrantyClaim(Base):
    __tablename__ = "warranty_claims"
    id = Column(Integer, primary_key=True)
    claim_id = Column(String(50), unique=True, nullable=False)
    claim_date = Column(Date, nullable=False, index=True)
    part_number = Column(String(50), nullable=False)
    part_description = Column(String(200))
    truck_vin = Column(String(17))
    build_date = Column(Date, nullable=True)
    plant = Column(String(20))
    supplier_id = Column(String(50), nullable=True)
    failure_mode = Column(String(100))
    repair_cost = Column(Float)
    is_systemic = Column(Boolean, default=False)
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class DemandActual(Base):
    __tablename__ = "demand_actuals"
    id = Column(Integer, primary_key=True)
    demand_date = Column(Date, nullable=False, index=True)
    part_number = Column(String(50), nullable=False)
    region = Column(String(50))
    qty_sold = Column(Float)
    revenue = Column(Float)
    channel = Column(String(30))    # dealer, direct, fleet
    source = Column(String(30), default="mock")
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyMetric(Base):
    __tablename__ = "daily_metrics"
    id = Column(Integer, primary_key=True)
    metric_date = Column(Date, nullable=False, unique=True, index=True)
    fill_rate_pct = Column(Float)
    open_backorders = Column(Integer)
    critical_backorders = Column(Integer)
    supplier_otd_pct = Column(Float)
    production_attainment_pct = Column(Float)
    freight_exception_count = Column(Integer)
    freight_on_time_pct = Column(Float)
    warranty_claims_count = Column(Integer)
    warranty_cost = Column(Float)
    inventory_value = Column(Float)
    active_pos = Column(Integer)
    po_overdue_count = Column(Integer)
    raw_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class WeeklyMetric(Base):
    __tablename__ = "weekly_metrics"
    id = Column(Integer, primary_key=True)
    week_start = Column(Date, nullable=False, unique=True, index=True)
    week_end = Column(Date)
    inventory_turns = Column(Float)
    forecast_mape = Column(Float)
    avg_fill_rate_pct = Column(Float)
    avg_supplier_otd_pct = Column(Float)
    avg_production_attainment_pct = Column(Float)
    total_freight_cost = Column(Float)
    total_warranty_cost = Column(Float)
    total_po_value = Column(Float)
    excess_inventory_value = Column(Float)
    short_inventory_parts = Column(Integer)
    top_warranty_part = Column(String(50), nullable=True)
    raw_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConnectorLog(Base):
    __tablename__ = "connector_logs"
    id = Column(Integer, primary_key=True)
    connector_name = Column(String(50))
    status = Column(String(20))     # success, error, skipped
    records_pulled = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer)
    run_at = Column(DateTime, default=datetime.utcnow)
