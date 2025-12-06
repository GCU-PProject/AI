# src/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.schemas.chat import ChatRequest, CommonResponse, ChatResult
from src.services.rag_service import generate_answer

router = APIRouter()


@router.post("/chat_test", response_model=CommonResponse)
def chat_endpoint(request: ChatRequest):
    # 지금은 RAG 구현 전이므로, 구조가 잘 잡혔는지 확인하는 가짜 응답만 보냅니다.
    return CommonResponse(
        isSuccess=True,
        code="COMMON200",
        message="API 구조 리팩토링 완료!",
        result={"echo": request.query},  # 받은 질문을 그대로 돌려줌 (테스트용)
    )


@router.post("/chat", response_model=CommonResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    법률 Q&A 챗봇 API (RAG)
    """
    try:
        # 서비스 로직 호출 (여기가 핵심!)
        result_data = await generate_answer(request.query, db)

        # Pydantic 모델로 변환
        chat_result = ChatResult(**result_data)

        return CommonResponse(
            isSuccess=True, code="AI200", message="성공입니다.", result=chat_result
        )

    except Exception as e:
        # 에러 발생 시 처리
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message=f"서버 내부 오류: {str(e)}",
            result=None,
        )
