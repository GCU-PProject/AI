# src/schemas/common.py
"""
공통 API 응답 스키마

모든 API가 동일한 형식으로 응답하기 위한 표준 응답 래퍼입니다.

[왜 공통 응답 형식이 필요한가?]
프론트엔드에서 모든 API 응답을 동일한 방식으로 처리할 수 있습니다.
성공/실패 여부, 에러 코드, 메시지 등을 항상 같은 위치에서 확인할 수 있어
코드가 일관되고 단순해집니다.

[응답 예시]
성공 시:
    { "isSuccess": true,  "code": "AI200", "message": "성공입니다.", "result": {...} }

실패 시:
    { "isSuccess": false, "code": "AI500", "message": "서버 내부 오류...", "result": null }
"""

from pydantic import BaseModel
from typing import Optional, Any


class CommonResponse(BaseModel):
    isSuccess: bool  # 요청 성공 여부
    code: str  # 응답 코드 (예: "AI200" 성공, "AI500" 서버 에러)
    message: str  # 사용자에게 표시할 메시지
    result: Optional[Any] = (
        None  # 실제 데이터 (ChatResult, CompareResult 등이 들어감, 에러 시 null)
    )
