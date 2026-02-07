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


config = MCPConfig()
