# src/main.py
from fastapi import FastAPI
from src.core.config import settings  # 설정 가져오기
from src.api.v1.endpoint import chat  # chat 라우터 가져오기

app = FastAPI(title="GLAW AI Backend", version="0.1.0")

# 라우터 등록 (이제 /api/v1/chat 주소가 생깁니다)
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


# 헬스 체크용 (팀장님 서버가 "살아있니?" 하고 찔러볼 주소)
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "AI Server is ready to serve!",
        "mode": "Internal API",
        "current_db": settings.DB_NAME,  # 잘 연결된 DB 이름 확인
    }
