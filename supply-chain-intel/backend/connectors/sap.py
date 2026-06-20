"""
SAP S/4HANA connector via OData v4 APIs.
Requires SAP_BASE_URL, SAP_CLIENT_ID, SAP_CLIENT_SECRET in .env.
"""
from datetime import datetime
import httpx
from backend.connectors.base import BaseConnector
from backend.config import settings


class SAPConnector(BaseConnector):
    name = "sap"

    def __init__(self):
        self._token: str | None = None
        self._token_expires: datetime | None = None

    def _get_token(self) -> str:
        if self._token and self._token_expires and datetime.utcnow() < self._token_expires:
            return self._token
        resp = httpx.post(
            f"{settings.SAP_BASE_URL}/sap/bc/sec/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.SAP_CLIENT_ID,
                "client_secret": settings.SAP_CLIENT_SECRET,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        from datetime import timedelta
        self._token_expires = datetime.utcnow() + timedelta(seconds=data.get("expires_in", 3600) - 60)
        return self._token

    def _get(self, path: str, params: dict = None) -> list[dict]:
        headers = {"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"}
        resp = httpx.get(f"{settings.SAP_BASE_URL}{path}", headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def pull_inventory(self, since: datetime) -> list[dict]:
        raw = self._get(
            "/sap/opu/odata4/sap/api_material_stock/srvd_a2x/sap/materialstock/0001/MaterialStock",
            params={"$filter": f"LastChangeDateTime ge {since.isoformat()}Z"},
        )
        return [
            {
                "part_number": r["Material"],
                "location": r["Plant"],
                "location_type": "plant",
                "qty_on_hand": float(r.get("MatlWrhsStkQtyInMatlBaseUnit", 0)),
                "unit_cost": float(r.get("MaterialBaseUnit", 0)),
                "source": "sap",
            }
            for r in raw
        ]

    def pull_supplier_orders(self, since: datetime) -> list[dict]:
        raw = self._get(
            "/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder",
            params={"$filter": f"CreationDate ge datetime'{since.strftime('%Y-%m-%dT%H:%M:%S')}'"},
        )
        return [
            {
                "po_number": r["PurchaseOrder"],
                "supplier_id": r["Supplier"],
                "part_number": r.get("Material", ""),
                "qty_ordered": float(r.get("OrderQuantity", 0)),
                "unit_cost": float(r.get("NetPriceAmount", 0)),
                "source": "sap",
            }
            for r in raw
        ]

    def pull_production_output(self, since: datetime) -> list[dict]:
        raw = self._get(
            "/sap/opu/odata/sap/API_PRODUCTION_ORDERS_SRV/A_ProductionOrder",
            params={"$filter": f"CreationDate ge datetime'{since.strftime('%Y-%m-%dT%H:%M:%S')}'"},
        )
        return [
            {
                "plant": r["ProductionPlant"],
                "model_family": r.get("Material", ""),
                "units_planned": int(r.get("PlannedQuantity", 0)),
                "units_built": int(r.get("ConfirmedYieldQuantity", 0)),
                "source": "sap",
            }
            for r in raw
        ]

    def pull_freight_shipments(self, since: datetime) -> list[dict]:
        return []   # SAP TM endpoint — implement when TM module available

    def pull_warranty_claims(self, since: datetime) -> list[dict]:
        return []   # SAP QM / CS module

    def pull_demand_actuals(self, since: datetime) -> list[dict]:
        return []   # SAP SD sales orders

    def pull_dealer_inventory(self, since: datetime) -> list[dict]:
        return []   # Dealer portal / CDK connector

    def health_check(self) -> bool:
        try:
            self._get_token()
            return True
        except Exception:
            return False
