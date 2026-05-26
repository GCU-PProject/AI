# src/api/endpoint/chat.py
"""
법률 Q&A API 엔드포인트

이 파일은 LangChain 기반 서비스를 호출하는 API 엔드포인트입니다.

[엔드포인트 목록]
POST /api/qna → 법률 Q&A (services/chat_service 호출)
"""

import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.utils import error_response
from src.core.database import get_db
from src.models import Country
from src.schemas.chat import ChatRequest, ChatResult
from src.schemas.common import CommonResponse
from src.services.chat_service import generate_answer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/qna", response_model=CommonResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    법률 Q&A 챗봇 API

    LangChain 기반 RAG 서비스를 호출합니다.
    기능: 질문 영어 번역, TOP_K 검색, LCEL 파이프라인
    """
    try:
        country = await db.execute(
            select(Country).where(Country.country_id == request.country_id)
        )

        if not country.scalar():
            logger.warning("존재하지 않는 국가 ID입니다: %s", request.country_id)
            return error_response(
                status=404,
                code="AI_COUNTRY_NOT_FOUND",
                message=f"존재하지 않는 국가 ID입니다: {request.country_id}",
            )

        start_time = time.perf_counter()

        result_data = await generate_answer(
            query=request.query,
            db=db,
            country_id=request.country_id,
            session_id=request.session_id,
        )

        elapsed_time = round(time.perf_counter() - start_time, 2)
        logger.info(
            "⏱️ [Q&A Latency] 질문: '%s' | 답변 생성 시간: %s초 소요",
            request.query,
            elapsed_time,
        )

        chat_result = ChatResult(**result_data)

        return CommonResponse(
            success=True,
            status=200,
            code="SUCCESS",
            message="성공입니다.",
            result=chat_result,
        )

    except (ConnectionError, SQLAlchemyError) as e:
        logger.error(f"데이터베이스 오류: {str(e)}")
        return error_response(
            status=503,
            code="AI_DB_CONNECTION_FAILED",
            message="데이터베이스 연결에 실패했습니다.",
        )

    except ValueError as e:
        logger.error(f"검색/번역 실패: {str(e)}")
        return error_response(
            status=400,
            code="AI_RETRIEVAL_FAILED",
            message="법률 검색 처리 중 오류가 발생했습니다.",
        )

    except Exception:
        logger.exception("서버 내부 오류가 발생했습니다.")
        return error_response(
            status=500,
            code="COMMON500",
            message="서버 내부 오류가 발생했습니다.",
        )
