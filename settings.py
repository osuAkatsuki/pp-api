from __future__ import annotations

from starlette.config import Config
from starlette.datastructures import Secret

cfg = Config(".env")

SERVER_PORT: int = cfg("SERVER_PORT", cast=int)
DB_DSN: Secret = cfg("DB_DSN", cast=Secret)