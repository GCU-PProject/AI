from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.core.config import settings

# (1) 임베딩 모델: 텍스트를 768차원의 숫자 배열(벡터)로 변환
# - 사용 모델: Google Vertex AI의 text-embedding-005
# - 용도: 사용자 질문을 벡터로 변환하여 DB의 법률 벡터와 거리 비교
# - 출력: 768개의 숫자로 구성된 배열 (예: [0.012, -0.034, 0.056, ...])
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-005",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
)


# (2) LLM (Large Language Model): 답변을 생성하는 AI 모델
# - 사용 모델: Google Gemini (gemini-2.5-flash)
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
    )
