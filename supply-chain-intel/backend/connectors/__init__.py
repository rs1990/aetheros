from backend.config import settings
from backend.connectors.mock import MockConnector
from backend.connectors.sap import SAPConnector

def get_connectors():
    if settings.USE_MOCK_DATA:
        return [MockConnector()]
    connectors = []
    if settings.SAP_CLIENT_ID:
        connectors.append(SAPConnector())
    if not connectors:
        connectors.append(MockConnector())
    return connectors
