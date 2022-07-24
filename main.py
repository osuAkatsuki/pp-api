#!/usr/bin/env python3.9
from fastapi import FastAPI

import logging
import uvicorn
import api
import services
import settings

def init_events(app: FastAPI) -> None:
    @app.on_event("startup")
    async def on_startup() -> None:
        await services.connect_services()

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await services.disconnect_services()


def init_api() -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)

    init_events(app)

    return app


app = init_api()

def main() -> int:
    uvicorn.run(
        "main:app",
        reload=True,
        log_level=logging.WARNING,
        server_header=False,
        date_header=False,
        host="127.0.0.1",
        port=settings.SERVER_PORT,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())