# src/api/v1/endpoints/chat.py
from fastapi import APIRouter
from src.schemas.chat import ChatRequest
from src.schemas.response import CommonResponse

router = APIRouter()


@router.post("/chat", response_model=CommonResponse)
def chat_endpoint(request: ChatRequest):
    # 지금은 RAG 구현 전이므로, 구조가 잘 잡혔는지 확인하는 가짜 응답만 보냅니다.
    return CommonResponse(
        isSuccess=True,
        code="COMMON200",
        message="API 구조 리팩토링 완료!",
        result={"echo": request.query},  # 받은 질문을 그대로 돌려줌 (테스트용)
    )
