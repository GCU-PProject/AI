# src/main.py
"""
FastAPI 앱의 진입점 (Entry Point)

이 파일은 FastAPI 애플리케이션을 생성하고,
API 라우터를 등록합니다.

[서버 실행 방법]
uvicorn src.main:app --reload
→ 이 명령어에서 'src.main:app'은 "src/main.py 파일의 app 변수"를 의미합니다.
→ --reload: 코드 변경 시 서버가 자동으로 재시작됩니다.

[API 구조]
/                  → 헬스 체크 (서버 상태 확인)
/api/qna           → 법률 Q&A (LangChain 기반)
/api/compare       → 법률 비교
/api/risk          → 리스크 카드 조회

[load_dotenv()가 import보다 먼저 오는 이유]
.env 파일의 환경변수(GCP 인증 정보 등)를 OS에 먼저 등록해야
이후 import되는 모듈들(config.py, chat_service.py 등)이
환경변수를 정상적으로 읽을 수 있습니다.
"""

from dotenv import load_dotenv

# .env 파일의 환경변수를 OS에 등록 (GCP_PROJECT_ID, DB 접속 정보, 인증키 경로 등)
# ※ 반드시 다른 모듈을 import하기 전에 호출해야 합니다!
load_dotenv()

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.endpoint import chat, compare, risk
from src.core.config import settings

logging.basicConfig(level=logging.INFO)

# FastAPI 앱 생성
# - title: Swagger 문서(/docs)에 표시되는 API 이름
# - version: API 버전 (운영 배포 시 관리용)
app = FastAPI(title="GLAW AI Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 라우터 등록
# =========================================================
# include_router(): 별도 파일에서 정의한 엔드포인트들을 앱에 연결합니다.
# - prefix: URL 앞에 붙는 경로 (예: /api/qna)
# - tags: Swagger 문서에서 그룹핑할 이름

# 라우터: /api/qna, /api/compare, /api/risk
app.include_router(chat.router, prefix="/ai", tags=["Chat API"])
app.include_router(compare.router, prefix="/ai", tags=["Compare API"])
app.include_router(risk.router, prefix="/ai", tags=["Risk API"])


# =========================================================
# 글로벌 예외 처리 (API 명세서 규격 준수)
# =========================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    FastAPI의 기본 422 Unprocessable Entity 에러를 가로채서,
    팀의 공동 API 명세서에 맞게 400 COMMON400 에러로 변환하여 응답합니다.
    """

    missing_fields = []
    for error in exc.errors():
        loc = error.get("loc", [])
        if len(loc) > 1:
            missing_fields.append(str(loc[-1]))
        else:
            missing_fields.append(str(loc[0]))

    fields_str = ", ".join(missing_fields)
    message = f"요청 처리 중 오류 : 필수 입력값({fields_str})을 확인해주세요."

    content = {
        "success": False,
        "status": 400,
        "code": "COMMON400",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": None,
    }

    return JSONResponse(status_code=400, content=content)


# =========================================================
# 헬스 체크 엔드포인트
# =========================================================
# 서버가 정상적으로 동작하는지 확인하기 위한 최소한의 엔드포인트입니다.
# 프론트엔드나 모니터링 도구가 이 주소를 주기적으로 호출하여 서버 상태를 확인합니다.
# 브라우저에서 http://127.0.0.1:8000/ 으로 접속하면 확인할 수 있습니다.


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "AI Server is ready to serve!",
        "mode": "Internal API",
        "current_db": settings.DB_NAME,
    }
