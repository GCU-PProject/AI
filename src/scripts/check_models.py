import os
import sys

# 1. 프로젝트 루트 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
)
try:
    print("Testing...")
    res = llm.invoke("hi")
    print("SUCCESS:", res.content)
except Exception as e:
    import traceback

    traceback.print_exc()
