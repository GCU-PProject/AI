import httpx
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings


# (1) 임베딩 클라이언트: 로컬 Qwen3 임베딩 서버를 HTTP로 호출
# - 데이터(DB)는 Qwen3-Embedding-0.6B로 임베딩되어 있으므로, 쿼리도 같은 모델로 임베딩해야 함
# - instruction 접두사 + mean pooling + L2 정규화는 임베딩 서버(embed_server.py)가 처리하므로
#   여기서는 순수 텍스트만 그대로 전달한다 (접두사 중복 부착 금지)
# - LangChain의 Embeddings 인터페이스를 상속하여 chat_service / RAGAS 등과 호환 유지
#   · embed_query(text)        : 단일 쿼리 임베딩 (RAG 검색용)
#   · embed_documents(texts)   : 다수 문서 임베딩 (RAGAS KnowledgeGraph 구성 등)
class RemoteEmbeddings(Embeddings):
    def __init__(self, url: str = settings.EMBEDDING_URL):
        self.url = url.rstrip("/")

    def embed_query(self, text: str) -> list[float]:
        resp = httpx.post(f"{self.url}/embed", json={"inputs": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 임베딩 서버는 inputs로 리스트를 받으면 배치로 처리해 벡터 리스트를 반환한다.
        resp = httpx.post(f"{self.url}/embed", json={"inputs": texts}, timeout=120)
        resp.raise_for_status()
        return resp.json()


embeddings = RemoteEmbeddings()


# (2) 답변 생성 LLM (Google Gemini, 모델명은 .env의 GCP_MODEL_NAME)
# - temperature=0: 법률 서비스 특성상 일관되고 결정적인 답변 우선
# - top_k=20 / top_p=0.7: temperature=0에서는 영향이 미미하지만 안전장치로 설정
def get_llm(max_output_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GCP_MODEL_NAME,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        vertexai=True,
        temperature=0,
        max_tokens=max_output_tokens,
        top_k=20,
        top_p=0.7,
        # [추론(thinking) 설정 — 모델 비교 실험 시 둘 중 하나만 활성화]
        # - thinking_budget=0   : 추론 완전 비활성화 (lite 계열 모델은 이 파라미터도 불필요)
        # - thinking_level="low": 추론 약하게 유지 (3-flash/3.1-pro/3.5-flash만 지원)
        # ※ 두 파라미터를 동시에 주면 안 됨. 미설정 시 기본값은 high.
        # thinking_budget=0,
        thinking_level="low",
    )
