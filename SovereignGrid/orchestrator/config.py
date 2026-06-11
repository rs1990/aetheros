"""Orchestrator configuration — environment-driven, no secrets in code."""

import os


class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # Job lifecycle
    JOB_LEASE_SECONDS: int = int(os.getenv("GRID_JOB_LEASE_SECONDS", "300"))
    JOB_MAX_ATTEMPTS: int = int(os.getenv("GRID_JOB_MAX_ATTEMPTS", "3"))
    REQUEUE_INTERVAL_SECONDS: int = int(os.getenv("GRID_REQUEUE_INTERVAL", "15"))
    SETTLE_CRON_MINUTE: int = int(os.getenv("GRID_SETTLE_MINUTE", "0"))  # hourly at :00

    # Dispatch
    DISPATCH_TIMEOUT_SECONDS: int = int(os.getenv("GRID_DISPATCH_TIMEOUT", "300"))

    # Auth
    NONCE_TTL_SECONDS: int = 60

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.SUPABASE_URL and self.SUPABASE_SERVICE_KEY)


settings = Settings()
