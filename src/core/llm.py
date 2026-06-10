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


# (2) LLM (Large Language Model): 답변을 생성하는 AI 모델
# - 사용 모델: Google Gemini (settings.GCP_MODEL_NAME, 예: gemini-3.5-flash)
# - 용도: 검색된 법률 조항을 근거로 사용자에게 자연어 답변을 생성
#
# [파라미터 설명]
# - temperature (0~1): LLM 응답의 무작위성(창의성) 조절
#     0 = 가장 확률 높은 단어만 선택 → 일관되고 정확한 답변 (법률 서비스에 적합)
#     1 = 다양한 단어를 선택 → 창의적이지만 예측 불가능한 답변
# - max_output_tokens: 생성할 답변의 최대 길이
# - top_k: 다음 단어 생성 시 확률 상위 K개의 후보만 고려
#     20 = 상위 20개 단어 중에서만 선택 → 이상한 단어가 선택될 가능성 차단
# - top_p (nucleus sampling): 누적 확률이 P에 도달할 때까지의 단어만 후보로 사용
#     0.7 = 확률 합이 70%가 될 때까지의 단어만 고려 → 신뢰도 높은 단어 위주 선택
#
# ※ temperature=0이면 항상 최고 확률 단어를 선택하므로 top_k, top_p의 실질적 영향은
#   미미하지만, 안전장치로 설정
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
        thinking_level="low",  # 속도 최적화: default(high) → low (품질 확인 후 조정)
    )
