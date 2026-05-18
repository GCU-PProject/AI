# src/api/utils.py

from fastapi.responses import JSONResponse
from src.schemas.common import CommonResponse

def error_response(status: int, code: str, message: str) -> JSONResponse:
    payload = CommonResponse(
        success=False,
        status=status,
        code=code,
        message=message,
        result=None,
    )
    return JSONResponse(status_code=status, content=payload.model_dump())