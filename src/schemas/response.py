# src/schemas/response.py
from pydantic import BaseModel
from typing import Optional, Any


class CommonResponse(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[Any] = None  # 여기엔 아무거나(Any) 들어갈 수 있음
