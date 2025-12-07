# src/services/rag_service.py
import os
from typing import List, Dict, Any
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, label
from src.core.models import Law  # Law 모델이 src/core/models에 있다고 가정
from src.core.config import settings


# 모델 로드 (함수 호출 시마다 로드하지 않도록 전역 변수 처리 고려 가능)
def get_models():

    # GCP 프로젝트 설정
    # settings.GCP_PROJECT_ID와 settings.GCP_LOCATION은 .env에서 읽어옴
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    # 모델 로드
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    model_name = settings.GCP_MODEL_NAME or "gemini-1.5-flash-001"
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

    # 2. [검색] DB에서 유사한 법률 조항 1개 찾기
    # Law 모델에는 embedding 컬럼이 pgvector.sqlalchemy.Vector 타입이라고 가정
    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(1)
    )

    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return {
            "answer": "데이터가 없습니다.",
            "related_law_id_list": [],
            "search_success": False,
        }

    # 가장 가까운 1개 선택
    top_law = rows[0][0]
    # top_distance = rows[0][1] # Threshold 로직은 나중에 추가

    # 4. [프롬프트]
    context_text = f"- [{top_law.law_title} {top_law.article_no}]: {top_law.content}\n"
    law_ids = [top_law.law_id]

    # 시스템 프롬프트 (페르소나 + 가드레일)
    prompt = f"""
    당신은 'Global Legal Assistant'라는 전문 법률 비서입니다.
    반드시 아래 제공된 [근거 자료]만을 바탕으로 사용자의 질문에 답변하세요.
    
    [지시 사항]
    1. 답변은 반드시 [근거 자료]에 명시된 내용으로만 구성하세요. (환각 방지)
    2. [근거 자료]와 관련 없는 내용은 절대 지어내지 마세요.
    3. 만약 [근거 자료]로 답할 수 없다면, "죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다."라고만 말하세요.
    4. 답변은 친절하고 명확한 전문가의 어조(~입니다)로 작성하세요.

    [근거 자료]
    {context_text}

    [사용자 질문]
    {query}

    [답변]
    """

    # 5. [생성] Gemini에게 답변 요청
    try:
        response = generative_model.generate_content(prompt)
        final_answer = response.text
    except Exception as e:
        print(f"❌ Gemini 호출 실패: {e}")
        raise e  # 에러를 다시 던져서 API 엔드포인트에서 처리하도록 함

    # 6. 결과 반환
    return {
        "answer": final_answer,
        "related_law_id_list": law_ids,
        "search_success": True,
    }
