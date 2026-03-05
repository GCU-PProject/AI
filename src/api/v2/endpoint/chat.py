# src/api/v2/endpoint/chat.py
"""
[v2 LangChain] API 엔드포인트 - 법률 Q&A 및 비교

이 파일은 v2(LangChain 기반) 서비스를 호출하는 API 엔드포인트입니다.

[v1 엔드포인트(api/v1/endpoint/chat.py)와의 차이]
- 엔드포인트 구조(URL, 요청/응답 형식)는 v1과 동일합니다.
- 내부적으로 호출하는 서비스만 다릅니다:
  v1: src.v1_services → Vertex AI SDK 직접 호출
  v2: src.v2_services → LangChain 프레임워크 사용

[요청/응답 스키마 공유]
v1과 v2는 동일한 schemas/ 폴더의 모델을 공유합니다.
→ 프론트엔드는 URL만 v1→v2로 바꾸면 나머지는 그대로 사용 가능

[엔드포인트 목록]
POST /api/v2/chat       → 법률 Q&A (v2_services/chat_service 호출)
POST /api/v2/compare    → 법률 비교 (v2_services/compare_service 호출)
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.database import get_db
from src.schemas.common import CommonResponse
from src.schemas.chat import ChatRequest, ChatResult
from src.schemas.compare import CompareRequest, CompareResult
from src.core.models import Country
from src.v2_services.chat_service import generate_answer
from src.v2_services.compare_service import compare_laws


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=CommonResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    [v2 LangChain] 법률 Q&A 챗봇 API

    v1과 요청/응답 형식은 동일하며, 내부적으로 LangChain 기반 서비스를 호출합니다.
    v2에서 추가된 기능: 질문 영어 번역, TOP_K 5개, LCEL 파이프라인
    """
    country = await db.execute(
        select(Country).where(Country.country_id == request.country_id)
    )

    if not country.scalar():
        return CommonResponse(
            isSuccess=False,
            code="AI404",
            message=f"존재하지 않는 국가 ID입니다: {request.country_id}",
            result=None,
        )

    try:
        result_data = await generate_answer(
            query=request.query,
            db=db,
            country_id=request.country_id,
            session_id=request.session_id,
        )

        chat_result = ChatResult(**result_data)

        return CommonResponse(
            isSuccess=True, code="AI200", message="성공입니다.", result=chat_result
        )

    except ConnectionError:
        return CommonResponse(
            isSuccess=False,
            code="AI503",
            message="데이터베이스 연결에 실패했습니다.",
            result=None,
        )

    except ValueError as e:
        return CommonResponse(
            isSuccess=False,
            code="AI400",
            message=f"요청 처리 중 오류 : {str(e)}",
            result=None,
        )

    except Exception as e:
        logger.exception("[v2] Chat endpoint error")
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message="서버 내부 오류가 발생했습니다.",
            result=None,
        )


@router.post("/compare", response_model=CommonResponse)
async def compare_endpoint(request: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    [v2 LangChain] 국가 간 법률 비교 API

    v1과 요청/응답 형식은 동일하며, 내부적으로 LangChain 기반 서비스를 호출합니다.
    v2에서 추가된 기능: 질문 영어 번역, JsonOutputParser 자동 파싱
    """

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

        return CommonResponse(
            isSuccess=False,
            code="AI404",
            message=f"존재하지 않는 국가 ID입니다: {missing}",
            result=None,
        )

    try:
        result_data = await compare_laws(
            query=request.query,
            country_id_1=request.country_id_1,
            country_id_2=request.country_id_2,
            db=db,
        )

        compare_result = CompareResult(**result_data)

        return CommonResponse(
            isSuccess=True,
            code="AI200",
            message="[v2] 비교 분석 성공입니다.",
            result=compare_result,
        )

    except ConnectionError:
        return CommonResponse(
            isSuccess=False,
            code="AI503",
            message="데이터베이스 연결에 실패했습니다.",
            result=None,
        )

    except ValueError as e:
        return CommonResponse(
            isSuccess=False,
            code="AI400",
            message=f"요청 처리 중 오류 : {str(e)}",
            result=None,
        )

    except Exception as e:
        logger.exception("[v2] Compare endpoint error")
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message=f"비교 분석 중 오류 발생: {str(e)}",
            result=None,
        )
