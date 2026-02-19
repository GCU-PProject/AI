from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 1. 기본 환경 변수 (타입을 지정하면 자동 검증됨)
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    # GCP 관련
    GCP_PROJECT_ID: str
    GCP_LOCATION: str
    GCP_MODEL_NAME: str
    GOOGLE_APPLICATION_CREDENTIALS: str

    # 2. SQLAlchemy용 URL (API 서버용)
    @property
    def DATABASE_URL(self) -> str:
        # db_check.py에서 sslmode=require를 쓰고 계시므로 여기에 포함시킵니다.
        # 기본적으로 동기(Sync) 방식인 psycopg2를 사용합니다.
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"

    # (참고) 나중에 FastAPI에서 비동기(Async)가 필요하면 이걸 쓰세요.
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?ssl=require"

    # 3. Psycopg2용 파라미터 (크롤러 스크립트용)
    @property
    def CRAWLER_DB_PARAMS(self) -> dict:
        return {
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_HOST,
            "port": self.DB_PORT,
            "dbname": self.DB_NAME,
            "sslmode": "require",  # 딕셔너리에도 SSL 옵션 추가
        }

    # .env 파일 위치 지정
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 알 수 없는 변수 무시


# 설정 인스턴스 생성
settings = Settings()
