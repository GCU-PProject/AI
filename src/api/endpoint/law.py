"""
법률 출처 조회 API 엔드포인트

[엔드포인트 목록]
POST /api/law-url → 법률 ID 목록으로 출처 URL 등 참조 정보 조회
                    (qna 응답의 related_law_id_list를 그대로 전달)
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.utils import error_response
from src.core.database import get_db
from src.schemas.common import CommonResponse
from src.schemas.law import LawUrlRequest
from src.services.law_service import get_law_urls

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/law-url", response_model=CommonResponse)
async def get_laws_endpoint(request: LawUrlRequest, db: AsyncSession = Depends(get_db)):
    """
    법률 ID 목록으로 출처 URL 등 참조 정보를 조회합니다.

    Q&A 응답의 related_law_id_list를 그대로 넘기면, 각 법률의
    law_type, article_no, section_title, 출처 URL을 입력 순서대로 반환합니다.
    """
    try:
        result_data = await get_law_urls(request.law_id_list, db)

        return CommonResponse(
            success=True,
            status=200,
            code="SUCCESS",
            message="법률 출처 조회 성공입니다.",
            result=result_data,
        )

    except (ConnectionError, SQLAlchemyError) as e:
        logger.error(f"데이터베이스 오류: {str(e)}")
        return error_response(
            status=503,
            code="AI_DB_CONNECTION_FAILED",
            message="데이터베이스 연결에 실패했습니다.",
        )

    except ValueError as e:
        logger.error(f"법률 조회 실패: {str(e)}")
        return error_response(
            status=400,
            code="AI_RETRIEVAL_FAILED",
            message="법률 조회 처리 중 오류가 발생했습니다.",
        )

    except Exception:
        logger.exception("서버 내부 오류가 발생했습니다.")
        return error_response(
            status=500,
            code="COMMON500",
            message="서버 내부 오류가 발생했습니다.",
        )
