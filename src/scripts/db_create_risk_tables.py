import asyncio
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import AsyncEngine

import src.models  # noqa: F401
from src.core.database import Base, engine


async def create_risk_tables(async_engine: AsyncEngine) -> None:
    """risk / risk_list 테이블만 안전하게 생성합니다 (drop 없음)."""
    metadata = Base.metadata
    target_names = ["risk", "risk_list"]
    target_tables = [metadata.tables[name] for name in target_names]

    print("Creating tables (checkfirst=True): risk, risk_list")
    async with async_engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: metadata.create_all(
                bind=sync_conn,
                tables=target_tables,
                checkfirst=True,
            )
        )
    print("Done. Existing tables are preserved.")


if __name__ == "__main__":
    asyncio.run(create_risk_tables(engine))
