# src/schemas/chat.py
from pydantic import BaseModel
from typing import Optional


# /api/v1/chat 엔드포인트로 들어오는 요청 데이터 구조
class ChatRequest(BaseModel):
    query: str
    # 요청 바디에 country_code를 포함하여 DB 검색 필터링에 사용할 수 있도록 합니다.
    country_code: Optional[str] = None  # 예: "KR", "US". Null 가능하도록 Optional 설정
