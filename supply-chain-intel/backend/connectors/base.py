from abc import ABC, abstractmethod
from datetime import datetime


class BaseConnector(ABC):
    name: str = "base"

    @abstractmethod
    def pull_inventory(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_supplier_orders(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_production_output(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_freight_shipments(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_warranty_claims(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_demand_actuals(self, since: datetime) -> list[dict]:
        pass

    @abstractmethod
    def pull_dealer_inventory(self, since: datetime) -> list[dict]:
        pass

    def health_check(self) -> bool:
        return True
