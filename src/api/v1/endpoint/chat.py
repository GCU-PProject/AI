# src/api/v1/endpoint/chat.py
"""
[v1] API 엔드포인트 - 법률 Q&A 및 비교

이 파일은 FastAPI의 라우터(Router)를 정의합니다.
라우터는 "이 URL로 요청이 오면, 이 함수를 실행해라"를 연결하는 역할입니다.

[엔드포인트 목록]
POST /api/v1/chat_test  → 테스트용 에코 (서비스 로직 없이 요청을 그대로 반환)
POST /api/v1/chat       → 법률 Q&A (v1_services/chat_service 호출)
POST /api/v1/compare    → 법률 비교 (v1_services/compare_service 호출)

[처리 흐름]
클라이언트 요청 → 엔드포인트 함수 → 서비스 로직 → 응답
                   (이 파일)        (v1_services/)

[v2 엔드포인트(api/v2/endpoint/chat.py)와의 차이]
구조는 거의 동일하며, 호출하는 서비스만 다릅니다.
- v1: src.v1_services.chat_service 호출
- v2: src.v2_services.chat_service 호출

[에러 처리 전략]
모든 엔드포인트에서 try-except로 서비스 호출을 감싸고,
에러 발생 시 CommonResponse(success=False, status=500, code="COMMON500")을 반환합니다.
→ 서버가 500 에러로 죽지 않고, 프론트엔드에 구조화된 에러 메시지를 전달합니다.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.schemas.common import CommonResponse
from src.schemas.chat import ChatRequest, ChatResult
from src.schemas.compare import CompareRequest, CompareResult
from src.v1_services.chat_service import generate_answer
from src.v1_services.compare_service import compare_laws

# APIRouter: FastAPI의 라우터 객체
# main.py에서 app.include_router(router, prefix="/api/v1")으로 등록하면
# 여기서 정의한 /chat, /compare 경로가 /api/v1/chat, /api/v1/compare가 됩니다.
router = APIRouter()

# 로거 설정: 에러 발생 시 터미널에 상세 로그를 출력하기 위해 사용
# __name__은 모듈 경로(예: "src.api.v1.endpoint.chat")가 됩니다.
logger = logging.getLogger(__name__)


# =========================================================
# 테스트용 엔드포인트 (서비스 로직 없이 요청을 그대로 반환)
# =========================================================
# 서버가 정상 동작하는지, 요청/응답 형식이 맞는지 확인하기 위한 용도입니다.


@router.post("/chat_test", response_model=CommonResponse)
def chat_endpoint_test(request: ChatRequest):
    return CommonResponse(
        success=True,
        status=200,
        code="COMMON200",
        message="API 구조 리팩토링 완료!",
        result={"echo": request.query},
    )


# =========================================================
# 법률 Q&A 엔드포인트
# =========================================================
# [Depends(get_db)의 동작 원리]
# FastAPI의 '의존성 주입(Dependency Injection)' 패턴입니다.
# 1. 요청이 들어오면 FastAPI가 자동으로 get_db()를 호출하여 DB 세션을 생성
# 2. 생성된 세션을 db 파라미터에 주입
# 3. 함수 실행이 끝나면 세션을 자동으로 닫음
# → 개발자가 세션 생성/정리 코드를 매번 작성할 필요가 없어집니다.


@router.post("/chat", response_model=CommonResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    법률 Q&A 챗봇 API

    - query: 사용자 질문 (예: "음주운전 처벌은?")
    - country_id: 국가 ID (예: 1: California, 2: New York)
    """
    try:
        # 서비스 레이어 호출: RAG 파이프라인 실행 (임베딩 → 검색 → LLM 호출)
        result_data = await generate_answer(
            query=request.query, db=db, country_id=request.country_id
        )

        # 딕셔너리를 Pydantic 모델로 변환 (응답 형식 검증)
        # ChatResult(**result_data): dict의 키를 ChatResult의 필드에 매핑
        chat_result = ChatResult(**result_data)

        # CommonResponse로 래핑하여 반환
        # 모든 API가 동일한 형식(isSuccess, code, message, result)으로 응답합니다.
        return CommonResponse(
            success=True,
            status=200,
            code="SUCCESS",
            message="성공입니다.",
            result=chat_result,
        )

    except Exception as e:
        # 에러 종류: Vertex AI 통신 오류, 임베딩 실패, DB 연결 오류 등
        # logger.exception(): 에러 메시지 + 전체 스택 트레이스를 로그에 기록
        logger.exception("Chat endpoint error")
        return CommonResponse(
            success=False,
            status=500,
            code="COMMON500",
            message="서버 내부 오류가 발생했습니다.",
            result=None,
        )


# =========================================================
# 법률 비교 엔드포인트
# =========================================================


@router.post("/compare", response_model=CommonResponse)
async def compare_endpoint(request: CompareRequest, db: AsyncSession = Depends(get_db)):
    """
    국가 간 법률 비교 API

    - query: 비교 질문 (예: "음주운전 처벌 비교해줘")
    - country_id_1: 기준 국가 ID (예: 1)
    - country_id_2: 비교 대상 국가 ID (예: 2)
    """
    try:
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

    except Exception as e:
        return CommonResponse(
            success=False,
            status=500,
            code="COMMON500",
            message=f"비교 분석 중 오류 발생: {str(e)}",
            result=None,
        )
