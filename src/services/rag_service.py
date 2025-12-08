# src/services/rag_service.py
import os
from typing import List, Dict, Any
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, label
from src.core.models import Law
from src.core.config import settings

# Top-K 설정: 가장 유사한 법률 3개를 가져옵니다
TOP_K = 3

# 임계값 설정: 거리(Distance)가 0.95 이하인 문서만 참고합니다.
# (0.95보다 크면 관련성이 없다고 판단하여 버립니다.)
MAX_DISTANCE_THRESHOLD = 0.95


# 모델 로드 (함수 호출 시마다 로드하지 않도록 전역 변수 처리 고려 가능)
def get_models():

    # GCP 프로젝트 설정
    # settings.GCP_PROJECT_ID와 settings.GCP_LOCATION은 .env에서 읽어옴
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    # 모델 로드
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    model_name = settings.GCP_MODEL_NAME
    generative_model = GenerativeModel(model_name)

    return embedding_model, generative_model


async def generate_answer(query: str, db: AsyncSession) -> Dict[str, Any]:
    embedding_model, generative_model = get_models()

    # 1. [임베딩] 질문을 벡터로 변환
    try:
        text_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = embedding_model.get_embeddings([text_input])
        query_vector = embeddings[0].values
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        # Vertex AI 통신 오류가 여기서 발생할 가능성이 높음
        raise e

    # 2. [검색] DB에서 유사한 법률 조항 Top-K개 찾기
    # Law 모델에는 embedding 컬럼이 pgvector.sqlalchemy.Vector 타입이라고 가정
    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    valid_docs = []
    related_ids = []

    # 검색된 모든 행(row)을 검사
    for row in rows:
        law = row[0]
        distance = row[1]

        # 임계값(0.95) 이내인 경우만 유효한 문서로 인정
        if distance <= MAX_DISTANCE_THRESHOLD:
            valid_docs.append(law)
            related_ids.append(law.law_id)

    # 유효한 문서가 하나도 없으면 바로 반환 (Gemini 호출 X)
    if not valid_docs:
        return {
            "answer": "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다. (관련도 낮음)",
            "related_law_id_list": [],
            "search_success": False,
        }

    # 여러 개의 문서를 하나의 컨텍스트로 합치기
    context_text = ""
    for law in valid_docs:
        context_text += f"""
        [문서 ID: {law.law_id}]
        - 법률명: {law.law_title}
        - 조항: {law.article_no}
        - 내용: {law.content}
        --------------------------------------------------
        """

    # 시스템 프롬프트 (페르소나 + 가드레일)
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

    # 5. [생성] Gemini에게 답변 요청
    try:
        # LLM 생성 파라미터 설정
        config = GenerationConfig(
            temperature=0.0,  # temperature : 0에 가까울수록 가장 확률 높은 단어를 선택하여 답변이 논리적이고 정해진 틀을 따릅니다.
            max_output_tokens=1024,  # max_output_tokens : 생성할 최대 토큰 수 (약 700~800단어 분량)
            top_k=20,  # top_k : 상위 20개 단어만 후보로 둠 (이상한 단어 차단)
            top_p=0.7,  # top_p : 상위 70% 확률 내에서만 선택 (신뢰도 확보)
        )
        response = generative_model.generate_content(prompt, generation_config=config)
        final_answer = response.text
    except Exception as e:
        print(f"❌ Gemini 호출 실패: {e}")
        raise e  # 에러를 다시 던져서 API 엔드포인트에서 처리하도록 함

    # 6. 결과 반환
    return {
        "answer": final_answer,
        "related_law_id_list": related_ids,
        "search_success": True,
    }
