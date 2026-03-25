# src/core/database.py
"""
비동기 데이터베이스 연결 설정

이 파일은 FastAPI 서버가 PostgreSQL 데이터베이스와
비동기(async) 방식으로 통신하기 위한 설정을 정의합니다.

[왜 비동기를 사용하는가?]
동기 방식: DB 쿼리 실행 중 서버가 멈추고 → 다른 요청을 처리 못함
비동기 방식: DB 쿼리 결과를 기다리는 동안 → 다른 요청을 처리할 수 있음
FastAPI는 비동기 서버이므로, DB 연결도 비동기 방식을 사용해야 성능이 유지됩니다.

[핵심 구성 요소]
1. engine: DB 서버와의 물리적 연결을 관리 (연결 풀)
2. AsyncSessionLocal: 개별 요청에서 사용할 DB 세션을 생성하는 팩토리
3. Base: ORM 모델(테이블 설계도)의 부모 클래스
4. get_db(): FastAPI의 의존성 주입 패턴으로 세션을 제공하는 함수
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from src.core.config import settings

# =========================================================
# 1. 비동기 엔진 생성
# =========================================================
# 엔진은 DB 서버와의 연결을 관리하는 핵심 객체입니다.
# 내부적으로 '연결 풀(Connection Pool)'을 유지하여
# 매 요청마다 새 연결을 만들지 않고 기존 연결을 재사용합니다.
#
# - ASYNC_DATABASE_URL: "postgresql+asyncpg://..." 형식의 비동기 접속 URL
# - echo=False: SQL 쿼리 로그를 출력하지 않음 (True로 바꾸면 디버깅용 로그 출력)
# - future=True: SQLAlchemy 2.0 스타일 사용
engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False, future=True)

# =========================================================
# 2. 세션 팩토리 생성
# =========================================================
# 세션(Session)은 DB와의 대화 단위입니다.
# 하나의 API 요청 동안 하나의 세션을 사용합니다.
#
# sessionmaker()는 세션을 생성하는 '공장(Factory)'을 만듭니다.
# AsyncSessionLocal()을 호출할 때마다 새 세션 객체가 생성됩니다.
#
# - bind=engine: 위에서 만든 엔진에 연결
# - class_=AsyncSession: 비동기 세션 사용
# - autoflush=False: 자동으로 DB에 반영하지 않음 (명시적 commit 필요)
# - autocommit=False: 자동 커밋하지 않음 (트랜잭션 직접 관리)
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, autoflush=False, autocommit=False
)

# =========================================================
# 3. ORM 모델 기본 클래스
# =========================================================
# models.py에서 정의하는 테이블 클래스들이 이 Base를 상속합니다.
# 예: class Law(Base): ...
# Base는 모든 모델 정보를 metadata에 모아두고,
# create_all() 호출 시 이 정보를 기반으로 실제 테이블을 생성합니다.
Base = declarative_base()


# =========================================================
# 4. DB 세션 의존성 함수 (FastAPI Depends 패턴)
# =========================================================
# FastAPI의 Depends()와 함께 사용하여,
# 각 API 요청에 자동으로 DB 세션을 제공합니다.
#
# [사용 예시]
# @router.post("/chat")
# async def chat_endpoint(db: AsyncSession = Depends(get_db)):
#     result = await db.execute(...)  # DB 쿼리 실행
#
# [yield의 역할]
# - yield 이전: 세션을 생성하여 API 함수에 전달
# - yield 이후 (함수가 끝나면): 세션을 자동으로 닫음 (정리)
# 이렇게 하면 세션을 직접 닫는 코드를 매번 작성할 필요가 없습니다.


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
