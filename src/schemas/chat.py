# src/schemas/chat.py
"""
법률 Q&A API의 요청/응답 스키마

- ChatRequest: POST /api/qna의 요청 본문
- ChatResult: CommonResponse의 result 필드에 들어가는 응답 데이터
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional


# ---------------------------------------------------
# [요청] Request - 클라이언트(프론트엔드)가 보내는 데이터
# ---------------------------------------------------
class ChatRequest(BaseModel):
    query: str  # 사용자의 법률 질문 (예: "음주운전 하면 면허가 정지되나요?")
    country_id: int  # 검색 대상 국가 ID (예: 2=California, 3=New York)

    # 대화 세션 ID (대화 맥락 기억 기능용)
    # - 프론트엔드에서 UUID를 생성하여 전달합니다.
    # - 같은 session_id로 요청하면 이전 대화를 기억합니다.
    # - None이면 맥락 없이 단독 질문으로 처리합니다. (기존 동작과 동일)
    session_id: Optional[str] = None

    @field_validator("query")
    def check_query(cls, v):
        if not v or not v.strip():
            raise ValueError("질문을 입력해주세요.")
        return v


# ---------------------------------------------------
# [응답] Response - 서버가 반환하는 데이터
# ---------------------------------------------------
# CommonResponse의 result 필드에 들어갑니다.
# 최종 응답 형태: { isSuccess: true, code: "AI200", message: "...", result: ChatResult }
class ChatResult(BaseModel):
    answer: str  # AI가 생성한 최종 답변 텍스트
    related_law_id_list: List[int]  # 답변에 사용된 법률 ID 목록 (예: [68722, 68724])
    search_success: bool  # 벡터 검색 성공 여부 (임계값 통과 문서가 있었는지)
