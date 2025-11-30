# src/main.py

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# GCP / Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel

# DB 관련 라이브러리 (비동기 PostgreSQL)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# -----------------------------
# .env 로드
# -----------------------------
load_dotenv()

# -----------------------------
# 환경 변수에서 GCP 설정 읽기
# -----------------------------
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GCP_MODEL_NAME", "gemini-2.5-flash-lite")

# -----------------------------
# 환경 변수에서 DB 설정 읽기
# -----------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    # 개발 단계 경고용 로그
    print("[WARNING] DB 환경변수가 일부 비어 있습니다. .env 설정을 확인해주세요.")

DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}" f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# -----------------------------
# SQLAlchemy 비동기 엔진/세션 설정
# -----------------------------
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # True로 하면 SQL 로그 출력
    future=True,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
)


# -----------------------------
# Vertex AI 초기화 함수
# -----------------------------
def init_vertex():
    """
    Vertex AI 초기화 함수.
    - GCP 프로젝트/리전 정보를 기반으로 vertexai.init() 호출
    """
    if not PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID 환경 변수가 설정되어 있지 않습니다.")

    vertexai.init(project=PROJECT_ID, location=LOCATION)


# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="GLAW AI Backend",
    version="0.1.0",
)


# -----------------------------
# 헬스 체크 엔드포인트
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# GCP 테스트용 요청 바디 스키마
# -----------------------------
class GcpTestRequest(BaseModel):
    prompt: str = "가천대에 대해 한줄 소개 해줘"


# -----------------------------
# GCP AI 연동 테스트 엔드포인트
# -----------------------------
@app.post("/gcp-test")
def gcp_ai_test(body: GcpTestRequest):
    """
    Vertex AI(Gemini) 모델 호출 테스트용 엔드포인트.
    - 입력: prompt (자연어 텍스트)
    - 출력: Gemini가 생성한 텍스트
    """
    try:
        # Vertex 초기화
        init_vertex()

        # Gemini 모델 객체 생성
        model = GenerativeModel(MODEL_NAME)

        # 프롬프트 전송
        response = model.generate_content(body.prompt)

        # 응답 텍스트 추출
        text_out = getattr(response, "text", str(response))

        return {
            "project_id": PROJECT_ID,
            "location": LOCATION,
            "model_name": MODEL_NAME,
            "prompt": body.prompt,
            "response": text_out,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# DB 연결 테스트 엔드포인트
# -----------------------------
@app.get("/db-test")
async def db_test():
    """
    Cloud SQL(PostgreSQL) 연결 및 pgvector 활성 상태 확인용 엔드포인트.

    반환:
    - 현재 접속된 DB 이름
    - PostgreSQL 버전 문자열
    - pgvector(extension 'vector') 활성 여부 (True/False)
    """
    try:
        async with AsyncSessionLocal() as session:
            # 현재 DB 이름 / 버전 확인
            result = await session.execute(
                text("SELECT current_database(), version();")
            )
            current_db, version = result.fetchone()

            # pgvector 확장 활성 여부 확인
            ext_result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            )
            has_vector = ext_result.fetchone() is not None

            return {
                "database": current_db,
                "version": version,
                "pgvector_enabled": has_vector,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
