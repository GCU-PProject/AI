import json
import vertexai
from typing import Dict, Any, List
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Law, Country
from src.core.config import settings

# ---------------------------------------------------------
# 1. 설정값 정의 (chat_service와 동일한 스타일)
# ---------------------------------------------------------
TOP_K = 3
MAX_DISTANCE_THRESHOLD = 1.5


# ---------------------------------------------------------
# 2. 모델 로드 헬퍼 (동일한 스타일)
# ---------------------------------------------------------
def get_models():
    # GCP 프로젝트 설정
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    # 모델 로드
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    model_name = settings.GCP_MODEL_NAME
    generative_model = GenerativeModel(model_name)

    return embedding_model, generative_model


# ---------------------------------------------------------
# 3. [내부 함수] 국가별 검색 로직 (중복 제거용)
# ---------------------------------------------------------
async def _search_laws(query_vector, country_id: int, db: AsyncSession) -> List[Law]:
    """특정 국가 ID로 벡터 검색을 수행하고 유효한 문서만 반환"""

    # DB 쿼리 (chat_service와 구조 동일)
    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id == country_id)
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    valid_docs = []
    for row in rows:
        law = row[0]
        distance = row[1]
        # 임계값 체크
        if distance <= MAX_DISTANCE_THRESHOLD:
            valid_docs.append(law)

    return valid_docs


# ---------------------------------------------------------
# 4. [메인] 법률 비교 서비스 로직
# ---------------------------------------------------------
async def compare_laws(
    query: str, db: AsyncSession, country_id_1: int, country_id_2: int
) -> Dict[str, Any]:

    embedding_model, generative_model = get_models()

    # 국가 정보 조회 (ID -> Name 변환용)
    # ---------------------------------------------------------
    # 두 국가의 ID로 DB를 한 번만 조회하여 {id: name} 맵을 만듭니다.
    country_stmt = select(Country).where(
        Country.country_id.in_([country_id_1, country_id_2])
    )
    country_result = await db.execute(country_stmt)
    countries = country_result.scalars().all()

    # 맵 생성 (예: {1: "KR", 2: "US"})
    country_map = {c.country_id: c.country_name for c in countries}
    # ---------------------------------------------------------

    # 1. [임베딩] 질문을 벡터로 변환 (1회만 수행)
    try:
        text_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = embedding_model.get_embeddings([text_input])
        query_vector = embeddings[0].values
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        raise e

    # 2. [검색] 두 국가 각각 검색 (Double Retrieval)
    # _search_laws 내부 로직은 chat_service의 검색 부분과 동일합니다.
    docs_1 = await _search_laws(query_vector, country_id_1, db)
    docs_2 = await _search_laws(query_vector, country_id_2, db)

    # 3. [검증] 데이터 존재 여부 확인
    if not docs_1 or not docs_2:
        missing_country = []

        # 데이터가 없는 경우, 위에서 만든 map을 이용해 국가 코드를 가져옴
        if not docs_1:
            # get(id, "Unknown")은 혹시 모를 ID 오류 방지용
            missing_country.append(country_map.get(country_id_1, str(country_id_1)))

        if not docs_2:
            missing_country.append(country_map.get(country_id_2, str(country_id_2)))

        # 메시지 생성 (예: "KR, US의 관련 법률 데이터를...")
        error_msg = (
            f"{', '.join(missing_country)}의 관련 법률 데이터를 찾을 수 없습니다."
        )

        return {
            "search_success": False,
            "country_1_result": {"related_law_ids": [], "summary": "자료 없음"},
            "country_2_result": {"related_law_ids": [], "summary": "자료 없음"},
            # 프론트엔드에 보여줄 에러 메시지
            "compare_summary": {"common": error_msg, "diff": ""},
        }

    # 4. [프롬프트] 컨텍스트 구성
    def format_context(docs):
        if not docs:
            return "(관련 법률 정보 없음)"

        context_text = ""
        for law in docs:
            context_text += f"""
            [문서 ID: {law.law_id}]
            - 법률명: {law.law_title}
            - 조항: {law.article_no}
            - 내용: {law.content}
            --------------------------------------------------
            """
        return context_text

    context_1 = format_context(docs_1)
    context_2 = format_context(docs_2)

    NO_DATA_MSG = "죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다."

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

    # 5. [생성] Gemini에게 답변 요청
    try:
        config = GenerationConfig(
            temperature=0.0,
            max_output_tokens=2048,
            response_mime_type="application/json",  # JSON 응답 강제
        )
        response = generative_model.generate_content(prompt, generation_config=config)

        # JSON 파싱
        analysis = json.loads(response.text)

    except Exception as e:
        print(f"❌ Gemini 호출/파싱 실패: {e}")
        # 에러 발생 시 기본값 채움
        analysis = {
            "summary_1": "분석 실패",
            "summary_2": "분석 실패",
            "common": "오류 발생",
            "diff": "오류 발생",
        }

    # 6. [반환] 결과 구조 조립
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
