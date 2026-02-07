import os
from typing import Optional
from pydantic_settings import BaseSettings


class MCPConfig(BaseSettings):
    server_name: str = "cortex-memory"
    server_version: str = "1.0.0"
    backend_api_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "MCP_",
        "env_file": ".env",
        "extra": "ignore"
    }


class BackboardConfig(BaseSettings):
    """Configuration for Backboard.io integration."""
    api_key: Optional[str] = None
    api_url: str = "https://app.backboard.io/api"
    enabled: bool = False
    eval_model: str = "gpt-4.1"

    # Guard settings
    guard_enabled: bool = True
    guard_threshold: float = 0.5
    guard_log_blocked: bool = True

    model_config = {
        "env_prefix": "BACKBOARD_",
        "env_file": ".env",
        "extra": "ignore"
    }

    @property
    def is_available(self) -> bool:
        """Check if Backboard is configured and available."""
        return bool(self.api_key) and (self.enabled or self.api_key is not None)

    @property
    def guard_active(self) -> bool:
        """Check if guard should be active."""
        return self.is_available and self.guard_enabled


config = MCPConfig()
backboard_config = BackboardConfig()
