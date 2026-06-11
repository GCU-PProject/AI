# src/schemas/common.py
"""
공통 API 응답 스키마 — 모든 API가 같은 형식으로 응답하기 위한 표준 래퍼

[응답 예시]
성공: { "success": true, "status": 200, "code": "SUCCESS", "message": "성공입니다.", "timestamp": "...", "result": {...} }
실패: { "success": false, "status": 500, "code": "COMMON500", "message": "서버 내부 오류...", "timestamp": "...", "result": null }
"""

from pydantic import BaseModel
from pydantic import Field
from typing import Optional, Any
from datetime import datetime, timezone


class CommonResponse(BaseModel):
    success: bool  # true/false
    status: int  # HTTP 상태 코드와 동일한 의미의 숫자 코드
    code: str  # 응답 코드 (예: "SUCCESS", "COMMON400")
    message: str  # 사용자에게 표시할 메시지
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    result: Optional[Any] = None  # 실제 데이터 (에러 시 null)
