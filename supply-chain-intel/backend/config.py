from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./supply_chain.db"
    SECRET_KEY: str = "change-me"
    USE_MOCK_DATA: bool = True

    SAP_BASE_URL: str = ""
    SAP_CLIENT_ID: str = ""
    SAP_CLIENT_SECRET: str = ""

    ARIBA_API_KEY: str = ""
    ARIBA_REALM: str = "PACCAR-T"

    BY_BASE_URL: str = ""
    BY_TOKEN: str = ""

    MANHATTAN_BASE_URL: str = ""
    MANHATTAN_TOKEN: str = ""

    CDK_BASE_URL: str = ""
    CDK_API_KEY: str = ""

    PACCAR_CONNECT_BASE_URL: str = ""
    PACCAR_CONNECT_CLIENT_ID: str = ""
    PACCAR_CONNECT_CLIENT_SECRET: str = ""

    SNOWFLAKE_ACCOUNT: str = ""
    SNOWFLAKE_USER: str = ""
    SNOWFLAKE_WAREHOUSE: str = "ANALYTICS_WH"
    SNOWFLAKE_DATABASE: str = "SUPPLY_CHAIN"

    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_RECIPIENTS: str = ""

    SLACK_WEBHOOK_URL: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
