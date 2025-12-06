from pydantic import BaseModel
from typing import List, Optional, Any


# [Request] 백엔드에서 받을 질문
class ChatRequest(BaseModel):
    query: str
    country_code: Optional[str] = None


# [Result] 성공 시 result 안에 들어갈 알맹이 데이터
class ChatResult(BaseModel):
    answer: str
    related_law_id_list: List[int]
    search_success: bool


# [Response] 최종 응답 껍데기 (팀 공통 포맷)
class CommonResponse(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[ChatResult] = None
