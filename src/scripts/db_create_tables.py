import asyncio
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import src.models  # noqa: F401
from src.core.database import Base, engine


async def create_tables() -> None:
    """Create all ORM tables if they do not already exist."""
    table_names = sorted(Base.metadata.tables.keys())
    print(f"Creating tables if missing: {', '.join(table_names)}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("Done. Existing tables are preserved.")


if __name__ == "__main__":
    asyncio.run(create_tables())
