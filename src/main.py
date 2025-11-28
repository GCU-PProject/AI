# src/main.py

from fastapi import FastAPI

from dotenv import load_dotenv

# .env 파일 로드 (.env에 GCP 관련 환경변수 저장해둔 상태)
load_dotenv()

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="GLAW AI Backend",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """
    서버 헬스 체크용 엔드포인트.
    배포 후 모니터링/ALB 헬스체크 등에도 사용할 수 있음.
    """
    return {"status": "ok"}
