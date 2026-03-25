# src/core/config.py
"""
환경변수 관리 (pydantic-settings)

이 파일은 .env 파일에서 환경변수를 읽어와 Python 객체로 관리합니다.

[왜 환경변수를 사용하는가?]
DB 비밀번호, GCP 인증키 경로 등 민감한 정보를 소스 코드에 직접 적으면
Git에 올라가 보안 문제가 발생합니다.
대신 .env 파일에 저장하고 (.gitignore로 Git 제외),
이 파일에서 읽어와 사용합니다.

[BaseSettings의 동작 원리]
pydantic-settings의 BaseSettings를 상속하면:
1. 클래스에 선언된 필드 이름과 같은 환경변수를 자동으로 찾아서 값을 채워줍니다.
   예: DB_USER 필드 → .env의 DB_USER=myuser 값을 자동으로 읽어옴
2. 타입을 지정하면 자동으로 변환 + 검증합니다.
   예: DB_PORT: int → 문자열 "5432"를 정수 5432로 자동 변환
3. 필수 필드에 값이 없으면 서버 시작 시 에러가 발생하여 조기에 문제를 발견할 수 있습니다.

[사용 방법]
from src.core.config import settings
print(settings.DB_HOST)  # .env의 DB_HOST 값 출력
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # =============================================
    # DB 접속 정보
    # =============================================
    # 이 값들은 .env 파일에서 자동으로 읽어옵니다.
    # 예: DB_USER=myuser → settings.DB_USER = "myuser"
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str  # 로컬 개발: localhost (SSH 터널링), 운영: Cloud SQL 내부 IP
    DB_PORT: int  # PostgreSQL 기본 포트: 5432
    DB_NAME: str

    # =============================================
    # GCP (Google Cloud Platform) 설정
    # =============================================
    GCP_PROJECT_ID: str  # GCP 프로젝트 ID
    GCP_LOCATION: str  # 리전 (예: us-central1)
    GCP_MODEL_NAME: str  # LLM 모델명 (예: gemini-2.0-flash)
    GOOGLE_APPLICATION_CREDENTIALS: str  # 서비스 계정 키 파일 경로 (예: keys/xxx.json)

    # =============================================
    # DB 접속 URL 생성 (property)
    # =============================================
    # @property: 메서드를 속성처럼 사용할 수 있게 해줍니다.
    # settings.DATABASE_URL 로 호출하면 URL 문자열이 반환됩니다.

    @property
    def DATABASE_URL(self) -> str:
        """
        동기(Sync) 방식 DB 접속 URL

        사용처: 크롤러 스크립트 등 동기 방식 코드
        드라이버: psycopg2 (PostgreSQL 기본 드라이버)
        형식: postgresql://user:password@host:port/dbname?sslmode=require
        """
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """
        비동기(Async) 방식 DB 접속 URL

        사용처: FastAPI 서버 (비동기 DB 연결)
        드라이버: asyncpg (비동기 전용 PostgreSQL 드라이버)
        형식: postgresql+asyncpg://user:password@host:port/dbname?ssl=require

        [동기 vs 비동기 URL의 차이]
        - 동기: postgresql://  (psycopg2 드라이버)
        - 비동기: postgresql+asyncpg://  (asyncpg 드라이버)
        - SSL 파라미터도 다름: sslmode=require vs ssl=require
        """
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?ssl=require"

    @property
    def CRAWLER_DB_PARAMS(self) -> dict:
        """
        크롤러 스크립트용 DB 접속 파라미터 (딕셔너리 형태)

        사용처: crawl_us_ca.py, load_to_db.py 등
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

    # .env 파일 설정
    class Config:
        env_file = ".env"  # 읽어올 .env 파일 경로
        env_file_encoding = "utf-8"  # 파일 인코딩
        extra = "ignore"  # .env에 Settings에 정의되지 않은 변수가 있어도 무시


import os

# 설정 인스턴스 생성 (모듈 로드 시 1회 실행)
# 다른 파일에서 from src.core.config import settings 로 가져다 씁니다.
settings = Settings()

# Google Cloud SDK 인증을 위해 환경변수에 명시적으로 등록합니다.
# pydantic-settings는 객체에만 값을 저장할뿐 os.environ에 주입하지 않기 때문입니다.
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
