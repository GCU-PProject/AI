# src/core/database.py
"""
비동기 데이터베이스 연결 설정 (FastAPI + PostgreSQL)

- engine: DB 연결 풀 관리
- AsyncSessionLocal: 요청 단위 세션 팩토리
- Base: ORM 모델 부모 클래스
- get_db(): FastAPI Depends용 세션 제공 함수
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from src.core.config import settings

# echo=True로 바꾸면 SQL 쿼리 로그 출력 (디버깅용)
engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False, future=True)

# autoflush/autocommit=False: 명시적 commit으로 트랜잭션 직접 관리
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, autoflush=False, autocommit=False
)

Base = declarative_base()


# FastAPI Depends용 — 요청마다 세션을 제공하고 종료 시 자동으로 닫는다
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
