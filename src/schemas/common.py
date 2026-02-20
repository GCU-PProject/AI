# src/schemas/common.py
from pydantic import BaseModel
from typing import Optional, Any


# ---------------------------------------------------
# [응답] Response (Result)
# ---------------------------------------------------
class CommonResponse(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: Optional[Any] = None  # 여기에 ChatResult나 CompareResult가 들어갑니다.
