# src/core/config.py
"""
환경변수 관리 (pydantic-settings)

.env의 값을 타입 검증과 함께 읽어온다. 필수 필드가 비어 있으면
서버 시작 시점에 에러가 나 조기에 발견된다.

사용: from src.core.config import settings → settings.DB_HOST
"""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # =============================================
    # DB 접속 정보
    # =============================================
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str  # 로컬 개발: localhost (SSH 터널링), 운영: DB 내부 IP
    DB_PORT: int
    DB_NAME: str

    # =============================================
    # GCP (Google Cloud Platform) 설정
    # =============================================
    GCP_PROJECT_ID: str
    GCP_LOCATION: str  # 리전 (예: us-central1)
    GCP_MODEL_NAME: str  # 답변 생성 LLM 모델명 (예: gemini-3.5-flash)
    GOOGLE_APPLICATION_CREDENTIALS: str  # 서비스 계정 키 파일 경로 (예: keys/xxx.json)

    # =============================================
    # GCP Cloud Translation 설정
    # =============================================
    GCP_TRANSLATION_LOCATION: str = "us-central1"
    GCP_TRANSLATION_MODEL: str = "general/translation-llm"

    # =============================================
    # CORS 설정
    # =============================================
    # 예: CORS_ALLOW_ORIGINS=https://app.example.com,https://www.example.com
    CORS_ALLOW_ORIGINS: str = ""

    # =============================================
    # RAG 검색 설정
    # =============================================
    RAG_TOP_K: int = 5
    RAG_MAX_DISTANCE_THRESHOLD: float = 0.90

    # =============================================
    # 임베딩 서버 설정
    # =============================================
    # 로컬 Qwen3 임베딩 서버 주소 (임베딩 전용 VM 내부 IP:포트)
    # .env에서 EMBEDDING_URL로 덮어쓸 수 있음
    EMBEDDING_URL: str = "http://10.0.1.5:8081"

    # =============================================
    # DB 접속 URL 생성 (property)
    # =============================================

    @property
    def DATABASE_URL(self) -> str:
        """동기 방식 DB 접속 URL (psycopg2 — 크롤러 등 동기 스크립트용)."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """비동기 방식 DB 접속 URL (asyncpg — FastAPI 서버용).

        동기와 드라이버 접두사(postgresql+asyncpg://)와
        SSL 파라미터(ssl=require vs sslmode=require)가 다름에 주의.
        """
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?ssl=require"

    @property
    def CRAWLER_DB_PARAMS(self) -> dict:
        """
        크롤러 스크립트용 DB 접속 파라미터 (딕셔너리 형태)

        사용처: data_crawl_us_ca.py, db_load_law_data.py 등
        psycopg2.connect(**settings.CRAWLER_DB_PARAMS) 형태로 사용
        """
        return {
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_HOST,
            "port": self.DB_PORT,
            "dbname": self.DB_NAME,
            "sslmode": "require",
        }

    @property
    def CORS_ORIGINS(self) -> list[str]:
        """
        CORS 허용 Origin 목록을 반환합니다.

        - 환경변수 CORS_ALLOW_ORIGINS를 쉼표(,)로 구분해 입력합니다.
        - 빈 값이면 로컬 개발용 기본 Origin을 반환합니다.
        """
        if not self.CORS_ALLOW_ORIGINS or not self.CORS_ALLOW_ORIGINS.strip():
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]

        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]

    # .env 파일 설정
    class Config:
        env_file = ".env"  # 읽어올 .env 파일 경로
        env_file_encoding = "utf-8"  # 파일 인코딩
        extra = "ignore"  # .env에 Settings에 정의되지 않은 변수가 있어도 무시


# 설정 인스턴스 생성 (모듈 로드 시 1회 실행)
# 다른 파일에서 from src.core.config import settings 로 가져다 씁니다.
settings = Settings()

# Google Cloud SDK 인증을 위해 환경변수에 명시적으로 등록합니다.
# pydantic-settings는 객체에만 값을 저장할뿐 os.environ에 주입하지 않기 때문입니다.
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        settings.GOOGLE_APPLICATION_CREDENTIALS
    )
