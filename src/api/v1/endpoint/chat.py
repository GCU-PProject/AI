# src/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.schemas.common import CommonResponse
from src.schemas.chat import ChatRequest, ChatResult
from src.schemas.compare import CompareRequest, CompareResult
from src.services.chat_service import generate_answer
from src.services.compare_service import compare_laws

router = APIRouter()


# ⚠️ 기존 테스트용 /chat 엔드포인트 (유지)
@router.post("/chat_test", response_model=CommonResponse)
def chat_endpoint_test(request: ChatRequest):
    return CommonResponse(
        isSuccess=True,
        code="COMMON200",
        message="API 구조 리팩토링 완료!",
        result={"echo": request.query},
    )


@router.post("/chat", response_model=CommonResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    법률 Q&A 챗봇 API
    - query: 사용자 질문
    - country_code: (선택) 국가 코드 (예: KR, US, GB)
    """
    try:
        result_data = await generate_answer(
            query=request.query, db=db, country_id=request.country_id
        )

        chat_result = ChatResult(**result_data)

        return CommonResponse(
            isSuccess=True, code="AI200", message="성공입니다.", result=chat_result
        )

    except Exception as e:
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message=f"서버 내부 오류: {str(e)}",
            result=None,
        )


@router.post("/compare", response_model=CommonResponse)
async def compare_endpoint(request: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    국가 간 법률 비교 API
    - query: 비교 질문 (예: "한국과 미국의 저작권법 차이는?")
    - country_id_1: 기준 국가 ID
    - country_id_2: 비교 대상 국가 ID
    """
    try:
        # 서비스 로직 호출: compare_laws (또는 작성하신 서비스 함수명)
        result_data = await compare_laws(
            query=request.query,
            country_id_1=request.country_id_1,
            country_id_2=request.country_id_2,
            db=db,
        )

        # Pydantic 모델로 변환
        compare_result = CompareResult(**result_data)

        return CommonResponse(
            isSuccess=True,
            code="AI200",
            message="비교 분석 성공입니다.",
            result=compare_result,
        )

    except Exception as e:
        return CommonResponse(
            isSuccess=False,
            code="AI500",
            message=f"비교 분석 중 오류 발생: {str(e)}",
            result=None,
        )
