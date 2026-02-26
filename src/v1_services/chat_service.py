# src/v1_services/chat_service.py
"""
[v1] Vertex AI SDK 직접 호출 방식의 법률 Q&A 서비스 (RAG)

이 파일은 LangChain 없이 Google Vertex AI SDK를 직접 호출하여
RAG(Retrieval-Augmented Generation) 파이프라인을 구현합니다.

[v2와의 차이점]
v2에서는 LangChain 프레임워크로 리팩토링되었습니다.
이 v1 코드는 비교 참고용으로 유지하고 있습니다.

| 구분       | v1 (이 파일)                       | v2 (LangChain)              |
|------------|------------------------------------|-----------------------------|
| 임베딩     | TextEmbeddingModel.from_pretrained | VertexAIEmbeddings          |
| LLM        | GenerativeModel                    | ChatVertexAI                |
| 프롬프트   | Python f-string 직접 조립          | ChatPromptTemplate          |
| 체인 연결  | 각 단계를 수동으로 순차 호출       | LCEL 파이프라인 (| 연산자)  |
| 모델 초기화| 매 요청마다 get_models() 호출      | 모듈 로드 시 1회 초기화      |
| 번역       | 없음 (한국어 질문을 그대로 임베딩) | 검색 전 영어로 번역          |

[전체 처리 흐름]
사용자 질문
    → 1단계: GCP 프로젝트 초기화 + 모델 로드
    → 2단계: 질문을 벡터(숫자 배열)로 변환 (임베딩)
    → 3단계: DB에서 벡터 유사도 기반으로 관련 법률 조항 검색
    → 4단계: 검색 결과를 f-string으로 프롬프트에 삽입
    → 5단계: Gemini에 프롬프트 전달하여 답변 생성
    → 6단계: API 응답 반환

[v1의 한계점]
- 매 요청마다 vertexai.init()과 모델 로드를 반복하여 비효율적
- 한국어 질문을 번역 없이 그대로 임베딩하여 영어 법률과의 검색 정확도가 낮음
- 임베딩/검색/프롬프트/LLM이 하나의 함수에 몰려있어 유지보수가 어려움
"""

import os
from typing import List, Dict, Any
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, label
from src.core.models import Law, Country
from src.core.config import settings

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================

# TOP_K: 벡터 검색에서 가져올 최대 문서 수
# - 질문과 가장 유사한 법률 조항을 상위 몇 개까지 가져올지 결정
# ※ v2에서는 5로 변경됨 (더 많은 관련 법률을 참고하기 위해)
TOP_K = 3

# MAX_DISTANCE_THRESHOLD: L2 거리 기반 유사도 임계값
# - 벡터 간 거리가 이 값 이하인 문서만 "관련 있음"으로 판단
# - 0에 가까울수록 질문과 문서가 의미적으로 유사
# - 이 값보다 큰 문서는 관련성이 낮다고 판단하여 제외
MAX_DISTANCE_THRESHOLD = 0.85


# =========================================================
# 2. 모델 로드 함수
# =========================================================
# [v1의 문제점]
# 이 함수는 매 API 요청마다 호출됩니다.
# vertexai.init()과 모델 로드를 매번 반복하므로 비효율적입니다.
#
# [v2에서의 개선]
# v2에서는 모듈 로드 시 한 번만 VertexAIEmbeddings, ChatVertexAI 객체를 생성하고
# 이후 모든 요청에서 재사용합니다.


def get_models():
    """
    GCP 프로젝트를 초기화하고 임베딩/생성 모델을 로드합니다.

    Returns:
        (embedding_model, generative_model) 튜플
        - embedding_model: 텍스트 → 벡터 변환용
        - generative_model: 답변 생성용 (Gemini)
    """
    # GCP 프로젝트 설정 (프로젝트 ID와 리전은 .env에서 읽어옴)
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    # 임베딩 모델 로드 (text-embedding-005, 768차원 벡터 출력)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    # 생성 모델(LLM) 로드 (Gemini)
    model_name = settings.GCP_MODEL_NAME
    generative_model = GenerativeModel(model_name)

    return embedding_model, generative_model


# =========================================================
# 3. [메인] RAG 답변 생성 함수
# =========================================================
# 이 함수가 API 엔드포인트에서 호출되는 최종 진입점입니다.
# 임베딩 → 검색 → 프롬프트 조립 → LLM 호출 → 응답 반환을
# 하나의 함수 안에서 순차적으로 처리합니다.


async def generate_answer(
    query: str, db: AsyncSession, country_id: int
) -> Dict[str, Any]:
    """
    사용자의 법률 질문에 대해 RAG 기반 답변을 생성합니다.

    [전체 흐름]
    모델 로드 → 임베딩 → 벡터 검색 → 임계값 필터링 → 프롬프트 조립 → LLM 호출 → 응답

    Args:
        query: 사용자의 원본 질문
        db: 비동기 DB 세션 (SQLAlchemy AsyncSession)
        country_id: 검색 대상 국가 ID

    Returns:
        {
            "answer": "AI가 생성한 답변 텍스트",
            "related_law_id_list": [68722, 68724],
            "search_success": True/False
        }
    """
    # 모델 로드 (매 요청마다 호출 - v1의 한계점)
    embedding_model, generative_model = get_models()

    target_country_id = country_id
    print(f"🌍 국가 필터링 적용: ID {country_id}")

    # ----- Step 1: 질문을 벡터로 변환 (임베딩) -----
    # TextEmbeddingInput: Vertex AI 임베딩 모델에 전달할 입력 객체
    # - text: 벡터로 변환할 텍스트
    # - task_type: "RETRIEVAL_QUERY"는 검색 질문용 임베딩을 생성
    #   (문서 저장용은 "RETRIEVAL_DOCUMENT"를 사용)
    #
    # ※ v1에서는 한국어 질문을 그대로 임베딩합니다.
    #   v2에서는 먼저 영어로 번역한 뒤 임베딩하여 검색 정확도를 높였습니다.
    try:
        text_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = embedding_model.get_embeddings([text_input])
        query_vector = embeddings[0].values  # 768차원 숫자 배열
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        raise e

    # ----- Step 2: DB에서 벡터 유사도 검색 -----
    # pgvector의 l2_distance()로 질문 벡터와 각 법률 벡터 간의 거리를 계산합니다.
    # L2 거리(유클리드 거리): 값이 작을수록 두 벡터가 의미적으로 유사
    #
    # 쿼리 흐름:
    # 1. Law 테이블에서 country_id가 일치하는 법률만 필터링
    # 2. 질문 벡터와의 L2 거리를 계산
    # 3. 거리가 가까운 순서로 정렬
    # 4. 상위 TOP_K(3)개만 가져옴
    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id == target_country_id)
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    valid_docs = []
    related_ids = []

    # ----- Step 3: 임계값 필터링 -----
    # TOP_K개를 가져왔더라도, 거리가 임계값(0.85)보다 크면
    # 관련성이 낮다고 판단하여 제외합니다.
    for row in rows:
        law = row[0]  # Law 객체 (법률 데이터)
        distance = row[1]  # L2 거리 값

        if distance <= MAX_DISTANCE_THRESHOLD:
            valid_docs.append(law)
            related_ids.append(law.law_id)

    # 유효한 문서가 하나도 없으면 LLM을 호출하지 않고 바로 반환
    # (불필요한 API 호출 방지 → 비용 절약)
    if not valid_docs:
        return {
            "answer": "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다. (관련도 낮음)",
            "related_law_id_list": [],
            "search_success": False,
        }

    # ----- Step 4: 프롬프트 조립 (f-string 방식) -----
    # 검색된 법률 문서들을 하나의 텍스트로 합칩니다.
    #
    # [v2와의 차이]
    # - v1: f-string으로 직접 조립. 문서 ID(law_id)가 프롬프트에 노출됨
    # - v2: ChatPromptTemplate + format_docs 함수로 분리. DB ID 제거, 법률 코드만 표시
    context_text = ""
    for law in valid_docs:
        context_text += f"""
        [문서 ID: {law.law_id}]
        - 법률 종류: {law.law_type}
        - 목차: {law.section_title}
        - 조항: {law.article_no}
        - 내용: {law.content}
        --------------------------------------------------
        """

    # 시스템 프롬프트 (AI의 역할 + 답변 규칙 + 가드레일)
    # - 가드레일: 할루시네이션(없는 내용 지어내기) 방지를 위한 규칙들
    # - f-string이므로 변수({context_text}, {query})가 즉시 치환됩니다.
    #
    # [v2와의 차이]
    # v2에서는 ChatPromptTemplate을 사용하여 프롬프트를 분리 관리하고,
    # 인용 방식도 개선했습니다 (본문에 법률 코드 대신 [참고 법령] 섹션으로 분리)
    prompt = f"""
    당신은 'Global Legal Assistant'입니다. 
    전 세계 법률 정보를 바탕으로 사용자에게 정확하고 신뢰할 수 있는 정보를 제공하는 법률 AI 전문가입니다.
    
    반드시 아래 제공된 [근거 자료]만을 바탕으로 답변을 작성하십시오. 외부 지식은 절대 사용하지 마십시오.

    [근거 자료]
    {context_text}

    [사용자 질문]
    {query}

    [답변 작성 가이드라인]
    1. **관련성 최우선 (중요):** 가장 먼저 [근거 자료]가 [사용자 질문]의 주제(국가, 법률 대상, 상황)와 일치하는지 판단하십시오.
    2. **무관한 자료 무시:** 만약 [근거 자료]가 질문과 관련이 없다면(예: 태국 법률 질문에 한국 법률 자료가 주어진 경우), 절대 근거 자료 내용을 요약하거나 설명하지 마십시오.
    3. **답변 불가 처리:** 질문에 대한 답을 [근거 자료]에서 찾을 수 없다면, 다른 설명 없이 **"죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다."**라고만 답변하십시오.
    4. **핵심 요약:** 답변이 가능한 경우, 장황하게 설명하지 말고 핵심 내용만 간결하게 요약하십시오.
    5. **분량 제한:** 전체 답변 길이는 **3~5문장 내외**로 작성하십시오. (가독성 중시)
    6. **근거 중심:** 답변의 모든 내용은 위 [근거 자료]에 있는 내용이어야 합니다. 없는 내용은 절대 지어내지 마십시오.
    7. **본문 내 인용:** 별도의 출처 리스트를 만들지 말고, 답변 문장 속에서 자연스럽게 근거를 밝히십시오. (예: "도로교통법 제44조에 따르면...")
    8. **논리적 종합:** 여러 법률 조항이 있다면 이를 종합하여 하나의 결론으로 도출하십시오.
    9. **언어:** 사용자의 질문이 한국어라면, 근거 자료가 영어일지라도 반드시 **자연스러운 한국어**로 번역하여 답변하십시오.
    10. **면책 조항:** 답변을 제공한 경우에만 마지막에 줄을 바꾸고 "※ 본 답변은 법률적 조언이 아니며 정보 제공을 목적으로 합니다."라는 문구를 포함하십시오. (답변 불가 시에는 생략)

    [답변 형식]
    - **질문에 답할 수 있는 경우:**
        - **결론:** (질문에 대한 답을 1문장으로 명확하게 제시)
        - **상세 내용:** (법률 조항을 근거로 핵심 내용을 2~4문장으로 요약 설명)
    
    - **질문에 답할 수 없는 경우:**
        "죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다."
    """

    # ----- Step 5: Gemini에게 답변 요청 -----
    # GenerationConfig: LLM의 답변 생성 방식을 제어하는 파라미터
    #
    # [파라미터 설명]
    # - temperature (0~1): 응답의 무작위성(창의성) 조절
    #     0 = 가장 확률 높은 단어만 선택 → 일관되고 정확한 답변
    #     1 = 다양한 단어를 선택 → 창의적이지만 예측 불가능
    # - max_output_tokens: 생성할 답변의 최대 길이 (토큰 단위, 약 700~800 한국어 글자)
    # - top_k: 다음 단어 생성 시 확률 상위 K개 후보만 고려
    #     20 = 상위 20개 단어 중에서만 선택 → 이상한 단어 차단
    # - top_p (nucleus sampling): 누적 확률이 P에 도달할 때까지의 단어만 후보로 사용
    #     0.7 = 확률 합이 70%가 될 때까지의 단어만 고려
    #
    # ※ temperature=0이면 top_k, top_p의 실질적 영향은 미미하지만, 안전장치로 설정
    try:
        config = GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
            top_k=20,
            top_p=0.7,
        )
        response = generative_model.generate_content(prompt, generation_config=config)
        final_answer = response.text
    except Exception as e:
        print(f"❌ Gemini 호출 실패: {e}")
        # 에러를 다시 던져서 API 엔드포인트의 except 블록에서 처리
        raise e

    # ----- Step 6: 결과 반환 -----
    return {
        "answer": final_answer,
        "related_law_id_list": related_ids,
        "search_success": True,
    }
