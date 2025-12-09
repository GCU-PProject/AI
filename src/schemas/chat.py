# src/schemas/chat.py
from pydantic import BaseModel
from typing import List


# ---------------------------------------------------
# [요청] Request
# ---------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    country_id: int  # 예: "1 : 한국", "2 : 영국", '3 : 싱가포르"


# ---------------------------------------------------
# [응답] Response (Result)
# ---------------------------------------------------
class ChatResult(BaseModel):
    answer: str
    related_law_id_list: List[int]
    search_success: bool
