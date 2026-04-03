import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from src.core.database import get_db
from src.schemas.common import CommonResponse
from src.schemas.risk import RiskRequest
from src.core.models import Country
from src.v2_services.risk_service import get_risk_cards
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    payload = CommonResponse(
        success=False,
        status=status,
        code=code,
        message=message,
        result=None,
    )
    return JSONResponse(status_code=status, content=payload.model_dump())

@router.post("/risk", response_model=CommonResponse)
async def get_risk_endpoint(request: RiskRequest, db: AsyncSession = Depends(get_db)):
    """
    [v2] 리스크 카드 조회 API
    """
    try:
        country = await db.execute(
            select(Country).where(Country.country_id == request.country_id)
        )
        if not country.scalar():
            return _error_response(
                status=404,
                code="AI_COUNTRY_NOT_FOUND",
                message=f"존재하지 않는 국가 ID입니다: {request.country_id}",
            )

        risk_result = await get_risk_cards(
            country_id=request.country_id,
            travel_purpose=request.travel_purpose,
            visa_type=request.visa_type,
            age_band=request.age_band,
            db=db,
        )

        if not risk_result:
            return _error_response(
                status=404,
                code="AI_RISK_NOT_FOUND",
                message="해당 조건에 일치하는 리스크 카드가 없습니다.",
            )

        return CommonResponse(
            success=True,
            status=200,
            code="SUCCESS",
            message="리스크 조회 성공입니다.",
            result=risk_result,
        )

    except (ConnectionError, SQLAlchemyError):
        return _error_response(
            status=503,
            code="AI_DB_CONNECTION_FAILED",
            message="데이터베이스 연결에 실패했습니다.",
        )

    except ValueError:
        return _error_response(
            status=400,
            code="AI_RETRIEVAL_FAILED",
            message="법률 검색 처리 중 오류가 발생했습니다.",
        )

    except Exception as e:
        logger.exception("서버 내부 오류가 발생했습니다.")
        return _error_response(
            status=500,
            code="COMMON500",
            message="서버 내부 오류가 발생했습니다.",
        )
