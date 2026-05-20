import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.utils import error_response
from src.core.database import get_db
from src.core.models import Country
from src.schemas.common import CommonResponse
from src.schemas.compare import CompareRequest, CompareResult
from src.services.compare_service import compare_laws

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/compare", response_model=CommonResponse)
async def compare_endpoint(request: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    국가 간 법률 비교 API

    LangChain 기반 서비스를 호출합니다.
    기능: 질문 영어 번역, JsonOutputParser 자동 파싱
    """

    try:
        country_results = await db.execute(
            select(Country).where(
                Country.country_id.in_([request.country_id_1, request.country_id_2])
            )
        )
        country_results = country_results.scalars().all()

        if len(country_results) != 2:
            country_list = {c.country_id for c in country_results}
            missing = []
            if request.country_id_1 not in country_list:
                missing.append(request.country_id_1)
            if request.country_id_2 not in country_list:
                missing.append(request.country_id_2)

            return error_response(
                status=404,
                code="AI_COMPARE_COUNTRY_NOT_FOUND",
                message=f"존재하지 않는 국가 ID입니다: {missing}",
            )

        result_data = await compare_laws(
            query=request.query,
            country_id_1=request.country_id_1,
            country_id_2=request.country_id_2,
            db=db,
        )

        compare_result = CompareResult(**result_data)

        return CommonResponse(
            success=True,
            status=200,
            code="SUCCESS",
            message="비교 분석 성공입니다.",
            result=compare_result,
        )

    except (ConnectionError, SQLAlchemyError):
        return error_response(
            status=503,
            code="AI_DB_CONNECTION_FAILED",
            message="데이터베이스 연결에 실패했습니다.",
        )

    except Exception as e:
        logger.exception(f"비교 분석 중 오류 발생: {str(e)}")
        return error_response(
            status=500,
            code="AI_COMPARE_ANALYSIS_FAILED",
            message=f"비교 분석 중 오류 발생",
        )
