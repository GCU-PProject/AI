import os
import sys

# 1. 프로젝트 루트 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
from langchain_google_vertexai import ChatVertexAI

MODEL_NAME = (
    "gemini-2.5-flash"  # 이곳을 "gemini-3.0-flash" 등으로 변경하여 테스트하세요.
)

llm = ChatVertexAI(
    model_name=MODEL_NAME,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
)
try:
    print(f"Testing Model: {MODEL_NAME} ...")
    res = llm.invoke("hi")
    print("SUCCESS:", res.content)
except Exception:
    import traceback

    traceback.print_exc()
