import asyncio
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import AsyncEngine
from src.core.database import Base, engine  # Assuming 'engine' is an AsyncEngine
from src.core.models import (
    Law,
)  # Import models to ensure they are registered with Base.metadata


async def init_db(async_engine: AsyncEngine):
    """
    Initializes the database: drops all existing tables and creates new ones
    defined in Base.metadata.
    """
    print("Initializing database...")
    async with async_engine.begin() as conn:
        print("Dropping all tables...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(init_db(engine))
