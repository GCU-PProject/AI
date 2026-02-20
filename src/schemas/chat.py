# src/schemas/chat.py
from pydantic import BaseModel


# /api/v1/chat 엔드포인트로 들어오는 요청 데이터 구조
class ChatRequest(BaseModel):
    query: str
    country_id: int  # 예: "1 : 한국", "2 : 영국", '3 : 싱가포르"
