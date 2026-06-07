import os
import sys

# 1. 프로젝트 루트 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.core.config import settings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
)
try:
    print("Testing Model: gemini-3.1-pro-preview ...")
    res = llm.invoke("hi")
    print("SUCCESS:", res.content)
except Exception:
    import traceback

    traceback.print_exc()
