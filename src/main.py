# src/main.py

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel

# -----------------------------
# .env 로드
# -----------------------------
load_dotenv()

# -----------------------------
# 환경 변수에서 GCP 설정 읽기
# -----------------------------
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_NAME = os.getenv("GCP_MODEL_NAME", "gemini-2.5-flash-lite")


def init_vertex():
    """
    Vertex AI 초기화 함수.
    - GCP 프로젝트/리전 정보를 기반으로 vertexai.init() 호출
    """
    if not PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID 환경 변수가 설정되어 있지 않습니다.")

    vertexai.init(project=PROJECT_ID, location=LOCATION)


# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="GLAW AI Backend",
    version="0.1.0",
)


# -----------------------------
# 헬스 체크 엔드포인트
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# 요청 바디 스키마
# -----------------------------
class GcpTestRequest(BaseModel):
    prompt: str = "가천대에 대해 한줄 소개 해줘"


# -----------------------------
# GCP AI 연동 테스트 엔드포인트
# -----------------------------
@app.post("/gcp-test")
def gcp_ai_test(body: GcpTestRequest):
    """
    Vertex AI(Gemini) 모델 호출 테스트용 엔드포인트.
    - 입력: prompt (자연어 텍스트)
    - 출력: Gemini가 생성한 텍스트
    """
    try:
        # Vertex 초기화
        init_vertex()

        # Gemini 모델 객체 생성
        model = GenerativeModel(MODEL_NAME)

        # 프롬프트 전송
        response = model.generate_content(body.prompt)

        # 응답 텍스트 추출
        text = getattr(response, "text", str(response))

        return {
            "project_id": PROJECT_ID,
            "location": LOCATION,
            "model_name": MODEL_NAME,
            "prompt": body.prompt,
            "response": text,
        }

    except Exception as e:
        # 에러 발생 시 500 에러로 전달 (프론트/BE에서 로그 확인 가능)
        raise HTTPException(status_code=500, detail=str(e))
