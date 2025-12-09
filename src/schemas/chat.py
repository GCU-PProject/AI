# src/schemas/chat.py
from pydantic import BaseModel


# /api/v1/chat 엔드포인트로 들어오는 요청 데이터 구조
class ChatRequest(BaseModel):
    query: str
    country_id: int  # 예: "1 : 한국", "2 : 영국", '3 : 싱가포르"


class CompareRequest(BaseModel):
    query: str  # 비교할 주제 (예: "음주운전 처벌")
    country_id_1: int  # 첫 번째 국가 ID (예: 1)
    country_id_2: int  # 두 번째 국가 ID (예: 2)
