import databases
import contextlib
import settings

database: databases.Database

ctx_stack: contextlib.AsyncExitStack = contextlib.AsyncExitStack()

async def connect_services() -> None:
    global database
    database = await ctx_stack.enter_async_context(
        databases.Database(str(settings.DB_DSN)),
    )


async def disconnect_services() -> None:
    await ctx_stack.aclose()