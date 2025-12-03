# src/main.py
from fastapi import FastAPI
from dotenv import load_dotenv
from src.api.v1.endpoint import chat  # chat 라우터 가져오기

load_dotenv()

app = FastAPI(title="GLAW AI Backend", version="0.1.0")

# 라우터 등록 (이제 /api/v1/chat 주소가 생깁니다)
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/")
def read_root():
    return {"status": "online", "message": "Global Legal Assistant AI is Running!"}
