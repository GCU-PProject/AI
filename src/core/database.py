# src/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from src.core.config import settings

# 엔진 생성
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

# 세션 생성기
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, autoflush=False, autocommit=False
)

# 테이블 설계도용 Base 클래스
Base = declarative_base()


# API에서 DB 세션을 쓰기 위한 의존성 함수 (Dependency)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
