# src/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db  # 👈 DB 세션 의존성 가져오기
from src.schemas.chat import ChatRequest
from src.schemas.response import (
    CommonResponse,
    ChatResult,
)  # 👈 ChatResult 스키마 가져오기
from src.services.rag_service import generate_answer  # 👈 RAG 서비스 함수 가져오기
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


# ⚠️ 기존 테스트용 /chat 엔드포인트 (유지)
@router.post("/chat_test", response_model=CommonResponse)
def chat_endpoint_test(request: ChatRequest):
    # RAG 구현 전이므로, 구조 확인용 가짜 응답만 반환
    return CommonResponse(
        isSuccess=True,
        code="COMMON200",
        message="API 구조 리팩토링 완료!",
        result={"echo": request.query},
    )


@router.post("/chat", response_model=CommonResponse)
async def chat_endpoint(
    request: ChatRequest, db: AsyncSession = Depends(get_db)  # 👈 DB 의존성 주입
):
    """
    법률 Q&A 챗봇 API (RAG)
    - query: 사용자 질문
    - country_code: (선택) 국가 코드 (예: KR, US, GB)
    """
    try:
        # 서비스 로직 호출: DB 세션을 generate_answer 함수에 전달
        result_data = await generate_answer(
            query=request.query, db=db, country_id=request.country_id
        )

        # Pydantic 모델로 변환 (ChatResult는 response.py에 정의되어 있어야 함)
        chat_result = ChatResult(**result_data)

        return CommonResponse(
            isSuccess=True, code="AI200", message="성공입니다.", result=chat_result
        )

    except Exception as e:
        # Vertex AI, 임베딩, DB 연결 등 모든 서버 내부 오류 처리
        logger.exception("Chat endpoint error")
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message="서버 내부 오류가 발생했습니다.",
            result=None,
        )
