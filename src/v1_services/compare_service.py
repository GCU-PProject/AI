# src/v1_services/compare_service.py
"""
[v1] Vertex AI SDK 직접 호출 방식의 법률 비교 서비스

이 파일은 두 국가의 법률을 비교 분석하는 서비스를 구현합니다.
LangChain 없이 Google Vertex AI SDK를 직접 호출하며,
검색 결과를 f-string으로 프롬프트에 조립하고 JSON 응답을 수동으로 파싱합니다.

[chat_service.py와의 관계]
chat_service와 동일한 임베딩/검색 로직을 사용하지만,
v1에서는 공통 함수를 분리하지 않아 코드가 중복됩니다.
v2에서는 chat_service의 retrieve_laws(), format_docs()를 재사용하여 중복을 제거했습니다.

[v2와의 차이점]
| 구분       | v1 (이 파일)                           | v2 (LangChain)                      |
|------------|----------------------------------------|--------------------------------------|
| 프롬프트   | f-string ({{ }} 이중 이스케이프 필요)  | ChatPromptTemplate (이스케이프 불필요)|
| JSON 파싱  | json.loads() 수동 파싱                 | JsonOutputParser (자동 파싱)          |
| JSON 강제  | response_mime_type="application/json"  | JsonOutputParser가 자동 처리          |
| 검색       | 자체 _search_laws() 구현               | chat_service의 retrieve_laws() 재사용 |
| 모델 초기화| 매 요청마다 get_models() 호출          | 모듈 로드 시 1회 초기화               |

[전체 처리 흐름]
사용자 질문
    → 1단계: 국가 정보 조회 (country_id → country_name)
    → 2단계: 질문을 벡터로 변환 (임베딩, 1회만 수행)
    → 3단계: 두 국가 각각 벡터 검색 (Double Retrieval)
    → 4단계: 검색 결과가 없는 국가가 있으면 에러 반환
    → 5단계: 검색 결과를 f-string으로 프롬프트에 조립
    → 6단계: Gemini에 JSON 형식 응답 요청
    → 7단계: JSON 파싱 후 API 응답 형식으로 반환
"""

import json
import vertexai
from typing import Dict, Any, List
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Law, Country
from src.core.config import settings

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
# chat_service.py와 동일한 설정값입니다.
# ※ v2에서는 chat_service에서 import하여 한곳에서 관리합니다.

# TOP_K: 벡터 검색에서 가져올 최대 문서 수
TOP_K = 3

# MAX_DISTANCE_THRESHOLD: L2 거리 기반 유사도 임계값
MAX_DISTANCE_THRESHOLD = 0.85


# =========================================================
# 2. 모델 로드 함수
# =========================================================
# chat_service.py의 get_models()와 동일한 함수입니다.
# v1에서는 코드가 중복되지만, v2에서는 chat_service에서 공통 객체를 재사용합니다.


def get_models():
    """GCP 프로젝트 초기화 및 임베딩/생성 모델 로드"""
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    model_name = settings.GCP_MODEL_NAME
    generative_model = GenerativeModel(model_name)

    return embedding_model, generative_model


# =========================================================
# 3. 벡터 검색 함수 (국가별)
# =========================================================
# 한 국가에 대한 벡터 검색을 수행하는 내부 함수입니다.
# compare_laws()에서 두 국가를 각각 검색할 때 코드 중복을 줄이기 위해 분리했습니다.
#
# [v2와의 차이]
# v2에서는 chat_service의 retrieve_laws()를 재사용하므로 이 함수가 필요 없습니다.
# 또한 v2의 retrieve_laws()는 결과를 LangChain Document로 변환하지만,
# v1의 이 함수는 Law 객체를 그대로 반환합니다.


async def _search_laws(query_vector, country_id: int, db: AsyncSession) -> List[Law]:
    """
    특정 국가의 법률에서 벡터 유사도 검색을 수행합니다.

    Args:
        query_vector: 질문의 임베딩 벡터 (768차원 숫자 배열)
        country_id: 검색 대상 국가 ID
        db: 비동기 DB 세션

    Returns:
        임계값을 통과한 Law 객체 리스트
    """
    # pgvector의 l2_distance()로 질문 벡터와 법률 벡터 간의 거리 계산
    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id == country_id)
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    # 임계값(0.85) 이하인 문서만 유효한 결과로 반환
    valid_docs = []
    for row in rows:
        law = row[0]  # Law 객체
        distance = row[1]  # L2 거리 값
        if distance <= MAX_DISTANCE_THRESHOLD:
            valid_docs.append(law)

    return valid_docs


# =========================================================
# 4. [메인] 법률 비교 서비스
# =========================================================
# 이 함수가 API 엔드포인트(/api/v1/compare)에서 호출되는 최종 진입점입니다.


async def compare_laws(
    query: str, db: AsyncSession, country_id_1: int, country_id_2: int
) -> Dict[str, Any]:
    """
    두 국가의 법률을 비교 분석합니다.

    [전체 흐름]
    모델 로드 → 국가 조회 → 임베딩 → 검색(×2) → 검증 → 프롬프트 조립 → LLM(JSON) → 파싱 → 응답

    Args:
        query: 사용자의 원본 질문
        db: 비동기 DB 세션
        country_id_1: 기준 국가 ID
        country_id_2: 비교 국가 ID

    Returns:
        비교 분석 결과 (각 국가 요약 + 공통점/차이점)
    """
    # 모델 로드 (매 요청마다 호출 - v1의 한계점)
    embedding_model, generative_model = get_models()

    # ----- Step 1: 국가 정보 조회 (ID → 이름 변환) -----
    # 에러 메시지에서 국가 이름을 표시하기 위해
    # DB에서 {country_id: country_name} 매핑을 미리 조회합니다.
    # .in_() 연산자로 두 국가를 한 번의 쿼리로 동시에 조회합니다.
    country_stmt = select(Country).where(
        Country.country_id.in_([country_id_1, country_id_2])
    )
    country_result = await db.execute(country_stmt)
    countries = country_result.scalars().all()

    # 예: {1: "United States (California)", 2: "United States (New York)"}
    country_map = {c.country_id: c.country_name for c in countries}

    # ----- Step 2: 질문을 벡터로 변환 (임베딩) -----
    # 질문 임베딩은 1회만 수행하고, 두 국가 검색에 모두 같은 벡터를 사용합니다.
    # ※ v1에서는 한국어 질문을 그대로 임베딩 (v2에서는 영어 번역 후 임베딩)
    try:
        text_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = embedding_model.get_embeddings([text_input])
        query_vector = embeddings[0].values
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        raise e

    # ----- Step 3: 두 국가 각각 벡터 검색 (Double Retrieval) -----
    # 같은 질문 벡터로 두 국가의 법률을 각각 검색합니다.
    docs_1 = await _search_laws(query_vector, country_id_1, db)
    docs_2 = await _search_laws(query_vector, country_id_2, db)

    # ----- Step 4: 검색 결과 검증 -----
    # 한쪽이라도 유효한 검색 결과가 없으면 비교가 불가능합니다.
    # LLM을 호출하지 않고 바로 에러 응답을 반환합니다.
    if not docs_1 or not docs_2:
        missing_country = []

        if not docs_1:
            # get(id, str(id)): 매핑에 없을 경우 ID 숫자를 대신 표시 (안전장치)
            missing_country.append(country_map.get(country_id_1, str(country_id_1)))
        if not docs_2:
            missing_country.append(country_map.get(country_id_2, str(country_id_2)))

        error_msg = (
            f"{', '.join(missing_country)}의 관련 법률 데이터를 찾을 수 없습니다."
        )

        return {
            "search_success": False,
            "country_1_result": {"related_law_ids": [], "summary": "자료 없음"},
            "country_2_result": {"related_law_ids": [], "summary": "자료 없음"},
            "compare_summary": {"common": error_msg, "diff": ""},
        }

    # ----- Step 5: 프롬프트 조립 (f-string 방식) -----

    # (5-1) 검색 결과를 텍스트로 변환하는 내부 함수
    def format_context(docs):
        """Law 객체 리스트를 프롬프트에 삽입할 텍스트로 변환"""
        if not docs:
            return "(관련 법률 정보 없음)"

        context_text = ""
        for law in docs:
            context_text += f"""
            [문서 ID: {law.law_id}]
            - 법률 종류: {law.law_type}
            - 목차: {law.section_title}
            - 조항: {law.article_no}
            - 내용: {law.content}
            --------------------------------------------------
            """
        return context_text

    context_1 = format_context(docs_1)
    context_2 = format_context(docs_2)

    # (5-2) 답변 불가 메시지 (상수로 분리하여 프롬프트에서 일관되게 사용)
    NO_DATA_MSG = "죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다."

    # (5-3) 비교 분석 프롬프트
    #
    # [f-string에서 {{ }} 이중 이스케이프가 필요한 이유]
    # f-string은 {변수}를 Python 변수로 치환합니다.
    # 그런데 JSON 형식에도 중괄호 { }가 사용됩니다.
    # f-string이 JSON의 { }를 변수로 인식하지 않도록,
    # {{ }}로 이스케이프하면 출력 시 { }로 변환됩니다.
    #
    # 예: f"{{\"key\": \"value\"}}" → 출력: {"key": "value"}
    #
    # [v2에서의 개선]
    # v2의 ChatPromptTemplate은 {변수명}만 치환하므로
    # 일반 { }를 이스케이프할 필요가 없어 프롬프트가 훨씬 읽기 쉽습니다.
    prompt = f"""
    당신은 'Global Legal Assistant'입니다.
    전 세계 법률 정보를 바탕으로 두 국가의 법률을 객관적으로 비교 분석하는 법률 AI 전문가입니다.

    반드시 아래 제공된 [근거 자료]만을 바탕으로 답변을 작성하십시오. 외부 지식은 절대 사용하지 마십시오.

    [근거 자료 1 (기준 국가)]
    {context_1}

    [근거 자료 2 (비교 국가)]
    {context_2}

    [사용자 질문]
    {query}

    [답변 작성 가이드라인]
    1. **주제 적합성 검증 (최우선 순위):** - 답변 작성 전, [사용자 질문]의 의도와 [근거 자료]의 핵심 주제가 일치하는지 반드시 대조하십시오.
       - **[검증 예시]**: 질문이 "감자튀김(음식)"인데 자료가 "도로교통법"인 경우, 또는 질문이 "살인죄(형법)"인데 자료가 "건축법"인 경우 등 **주제가 논리적으로 무관하다면 절대 내용을 요약하지 마십시오.**
       - 주제가 불일치하거나 정보가 부족한 경우, 해당 필드에 **"{NO_DATA_MSG}"** 만을 입력하고 다음 단계로 넘어가지 마십시오.

    2. **답변 불가 메시지 처리 (엄격):** - 위 1번 검증 결과 답변이 불가능하다고 판단되면, **반드시 지정된 문구("{NO_DATA_MSG}")를 토씨 하나 바꾸지 말고 그대로 출력**하십시오. 
       - 이유를 설명하거나(예: "자료가 없어..."), 다른 말로 변형하지 마십시오.

    3. **핵심 요약:** - 주제가 일치하는 경우에만 수행하십시오.
       - 각 국가의 법률 내용은 핵심만 간결하게 **3~5문장 내외**로 요약하십시오.

    4. **비교 분석:** - 두 국가의 공통점(common)과 차이점(diff)을 명확한 논리로 도출하십시오. 
       - 만약 정보 부족으로 비교할 수 없다면, common과 diff 필드에도 위 **지정 문구**를 입력하십시오.

    5. **근거 중심:** 없는 내용은 절대 지어내지 말고, 문장 속에서 근거를 밝히십시오. (예: "제44조에 따르면...")

    6. **언어:** 근거 자료의 언어와 상관없이 반드시 **자연스러운 한국어**로 작성하십시오.

    7. **형식:** 반드시 아래 JSON 포맷을 준수하십시오. (마크다운 코드 블록 없이 순수 JSON만 출력)

    {{
        "summary_1": "기준 국가 법률 요약 (또는 주제 불일치/정보 부족 시 '{NO_DATA_MSG}' 출력)",
        "summary_2": "비교 국가 법률 요약 (또는 주제 불일치/정보 부족 시 '{NO_DATA_MSG}' 출력)",
        "common": "공통점 분석 (또는 주제 불일치/정보 부족 시 '{NO_DATA_MSG}' 출력)",
        "diff": "차이점 분석 (또는 주제 불일치/정보 부족 시 '{NO_DATA_MSG}' 출력)"
    }}
    """

    # ----- Step 6: Gemini에게 JSON 형식 답변 요청 -----
    # [response_mime_type 설명]
    # "application/json"을 지정하면 Gemini가 반드시 JSON 형식으로 응답합니다.
    # 이렇게 하면 마크다운 코드 블록(```json ... ```) 없이 순수 JSON만 출력되어
    # json.loads()로 바로 파싱할 수 있습니다.
    #
    # [v2와의 차이]
    # v2에서는 JsonOutputParser가 자동으로 JSON을 추출하므로
    # response_mime_type을 설정할 필요가 없습니다.
    try:
        config = GenerationConfig(
            temperature=0.0,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
        response = generative_model.generate_content(prompt, generation_config=config)

        # LLM 응답 텍스트를 Python dict로 변환 (수동 파싱)
        analysis = json.loads(response.text)

    except Exception as e:
        print(f"❌ Gemini 호출/파싱 실패: {e}")
        # JSON 파싱 실패 또는 API 호출 실패 시 기본값으로 대체
        analysis = {
            "summary_1": "분석 실패",
            "summary_2": "분석 실패",
            "common": "오류 발생",
            "diff": "오류 발생",
        }

    # ----- Step 7: 결과 반환 -----
    # LLM의 JSON 응답에서 각 필드를 추출하여 API 응답 형식에 맞게 재조립합니다.
    # analysis.get("key", ""): 키가 없을 경우 빈 문자열을 기본값으로 사용 (안전장치)
    return {
        "search_success": True,
        "country_1_result": {
            "related_law_ids": [law.law_id for law in docs_1],
            "summary": analysis.get("summary_1", ""),
        },
        "country_2_result": {
            "related_law_ids": [law.law_id for law in docs_2],
            "summary": analysis.get("summary_2", ""),
        },
        "compare_summary": {
            "common": analysis.get("common", ""),
            "diff": analysis.get("diff", ""),
        },
    }
