# src/services/chat_service.py
"""
LangChain 기반 법률 Q&A 서비스 (RAG 파이프라인)

[처리 흐름]
한국어 질문 → 영어 번역 → 임베딩 → 벡터 검색 → 검색 결과 + 원본 질문으로 답변 생성
"""
import asyncio
import html
import logging
import re
from typing import Any, Dict, List, Optional

from google.cloud import translate_v3
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.llm import embeddings, get_llm
from src.models import Country, Law
from src.services.memory import contextualize_question, save_to_history

logger = logging.getLogger(__name__)

# =========================================================
# 1. 설정값 (.env로 관리, 평가 실험으로 튜닝)
# =========================================================
# TOP_K: 벡터 검색에서 가져올 최대 문서 수 (기본값 5)
TOP_K = settings.RAG_TOP_K

# MAX_DISTANCE_THRESHOLD: L2 거리 임계값 (기본값 0.90)
# 이 거리보다 먼 문서는 관련성이 낮다고 판단해 제외합니다.
MAX_DISTANCE_THRESHOLD = settings.RAG_MAX_DISTANCE_THRESHOLD

# =========================================================
# 2. AI 모델 초기화 (모듈 로드 시 1회 생성, 전 요청 재사용)
# =========================================================
llm = get_llm()

# =========================================================
# 3. 질문 번역 (검색 정확도 향상을 위한 전처리)
# =========================================================
# 법률 데이터가 영어이므로 한국어 질문 그대로 임베딩하면 검색 정확도가 떨어진다.
# → 번역본은 검색에만 사용하고, 답변 생성에는 원본 한국어 질문을 사용한다.
# 번역 모델(Cloud Translation LLM)을 생성 모델과 분리해,
# 생성 모델 비교 실험에서도 검색 조건이 고정되도록 한다.

translation_client = translate_v3.TranslationServiceClient()
TRANSLATION_PARENT = (
    f"projects/{settings.GCP_PROJECT_ID}/locations/{settings.GCP_TRANSLATION_LOCATION}"
)
TRANSLATION_MODEL_PATH = (
    f"{TRANSLATION_PARENT}/models/{settings.GCP_TRANSLATION_MODEL}"
)
HANGUL_PATTERN = re.compile(r"[가-힣]")


def _translate_query_sync(query: str) -> str:
    """Cloud Translation 동기 클라이언트로 한국어 질문을 영어로 번역합니다."""
    response = translation_client.translate_text(
        request={
            "parent": TRANSLATION_PARENT,
            "contents": [query],
            "mime_type": "text/plain",
            "source_language_code": "ko",
            "target_language_code": "en",
            "model": TRANSLATION_MODEL_PATH,
        }
    )
    if not response.translations:
        raise ValueError("번역 결과가 없습니다.")
    return html.unescape(response.translations[0].translated_text)


@traceable
async def translate_query(query: str) -> str:
    """
    사용자의 질문을 영어로 번역합니다. (벡터 검색 정확도 향상 목적)

    - 이미 영어인 경우 그대로 반환합니다.
    - 번역 결과는 검색에만 사용되며, 최종 답변 생성에는 원본 질문이 사용됩니다.
    """
    if not query or not query.strip():
        raise ValueError("번역할 질문이 없습니다.")

    if not HANGUL_PATTERN.search(query):
        return query.strip()

    translated = await asyncio.to_thread(_translate_query_sync, query)

    if not translated or not translated.strip():
        raise ValueError("번역 결과가 없습니다.")
    logger.info("번역 완료: '%s' → '%s'", query, translated)
    return translated.strip()


# =========================================================
# 4. RAG 프롬프트 템플릿 (답변 생성용)
# =========================================================
# 설계 핵심: ① 근거 자료만 사용 + 관련 없으면 답변 불가 (할루시네이션 가드레일)
# ② 결론→상세→참고 법령→면책 순의 형식 통일 ③ 법률 코드는 [참고 법령] 섹션에만 표기

chat_yaml = load_prompt("src/prompts/chat.yaml", encoding="utf-8")

# {context}: 검색된 법률 텍스트 / {question}: 원본 질문 (한국어 그대로 → 한국어 답변 유도)
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", chat_yaml.template),
        ("human", "{question}"),
    ]
)


# =========================================================
# 5. 벡터 검색 함수 (커스텀 Retriever)
# =========================================================
# LangChain 내장 PGVector는 자체 테이블 구조를 요구하므로,
# 기존 laws 테이블을 그대로 쓰기 위해 SQLAlchemy 쿼리를 직접 작성한다.
# 검색 결과는 LangChain 표준 형식인 Document로 변환해 체인과 호환시킨다.


@traceable
async def resolve_jurisdiction_ids(country_id: int, db: AsyncSession) -> List[int]:
    """
    검색 대상 country_id 목록을 구성합니다.

    [목적]
    사용자가 특정 주(예: 캘리포니아)를 선택하면, 그 주의 법(주법)만이 아니라
    같은 국가의 연방법도 함께 검색해야 합니다. (연방법 + 주법 통합 검색)

    [규칙]
    - countries 테이블에서 state_code가 NULL인 행 = 연방(국가 단위)
    - 사용자가 주(state_code 있음)를 선택하면 → [선택한 주 + 같은 country_code의 연방]
    - 사용자가 연방(state_code 없음)을 선택하면 → [연방] 단독

    예) 캘리포니아(2, US/CA) 선택 → [2, 1]  (캘리포니아 주법 + 미국 연방법)
    """
    row = (
        await db.execute(select(Country).where(Country.country_id == country_id))
    ).scalar_one_or_none()

    if row is None:
        return [country_id]

    ids = [country_id]

    # 주(state)를 선택한 경우, 같은 국가의 연방(state_code IS NULL)을 추가
    if row.state_code is not None:
        federal_id = (
            await db.execute(
                select(Country.country_id).where(
                    Country.country_code == row.country_code,
                    Country.state_code.is_(None),
                )
            )
        ).scalar_one_or_none()
        if federal_id and federal_id != country_id:
            ids.append(federal_id)

    return ids


@traceable
async def retrieve_laws(
    query: str, country_id: int, db: AsyncSession
) -> tuple[List[Document], List[int]]:
    """
    벡터 유사도 기반으로 관련 법률 조항을 검색합니다.

    [처리 과정]
    1. 질문 텍스트를 1024차원 벡터로 변환 (임베딩)
    2. 법률 벡터와 L2 거리(유클리드 거리) 계산 (HNSW 인덱스 사용)
    3. 지정된 국가(country_id)의 법률 중 거리가 가장 가까운 TOP_K개 조회
    4. 임계값(MAX_DISTANCE_THRESHOLD) 이하인 문서만 유효한 결과로 반환

    Args:
        query: 검색할 질문 (영어로 번역된 상태)
        country_id: 검색 대상 국가 ID (예: 2=캘리포니아, 3=뉴욕)
        db: 비동기 DB 세션 (SQLAlchemy AsyncSession)

    Returns:
        (documents, law_ids) 튜플
        - documents: LangChain Document 리스트 (프롬프트에 삽입할 법률 텍스트)
        - law_ids: 검색된 법률의 ID 리스트 (API 응답의 related_law_id_list에 사용)
    """
    # Step 1: 질문을 벡터로 변환
    if not query:
        logger.warning("⚠️ retrieve_laws에 빈 query가 전달되었습니다.")
        return [], []

    query_str = str(query).strip()
    if not query_str:
        logger.warning("⚠️ retrieve_laws에 공백만 있는 query가 전달되었습니다.")
        return [], []

    # 질문을 1024차원 벡터로 변환 (Qwen3-Embedding-0.6B)
    try:
        query_vector = embeddings.embed_query(query_str)
    except Exception as e:
        logger.error("❌ embed_query 중 오류 발생! 입력 쿼리: %r, 에러: %s", query_str, e)
        raise

    # Step 2: 벡터 유사도 검색 — L2 거리가 가까운 순으로 TOP_K개 조회
    # 연방법 + 주법 통합 검색: 선택한 country_id를 [주 + 연방]으로 확장
    jurisdiction_ids = await resolve_jurisdiction_ids(country_id, db)
    logger.info("검색 대상 country_id 목록: %s", jurisdiction_ids)

    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id.in_(jurisdiction_ids))
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    documents = []
    law_ids = []

    for row in rows:
        law = row[0]
        distance = row[1]  # L2 거리 (0에 가까울수록 유사)

        # Step 3: 임계값 필터링 — 임계값보다 먼 문서는 관련성 낮음으로 제외
        if distance <= MAX_DISTANCE_THRESHOLD:
            # Step 4: LangChain Document로 변환
            doc = Document(
                page_content=law.content,
                metadata={
                    "law_id": law.law_id,
                    "law_type": law.law_type,
                    "section_title": law.section_title or "",
                    "article_no": law.article_no,
                    "distance": distance,
                },
            )
            documents.append(doc)
            law_ids.append(law.law_id)

    return documents, law_ids


# =========================================================
# 6. 컨텍스트 포맷 함수
# =========================================================


def format_docs(docs: List[Document]) -> str:
    """검색된 Document들을 프롬프트 삽입용 텍스트로 변환합니다.

    포맷: [법률코드 조항번호] + 목차 + 내용, 문서 간 구분선.
    예: "[VEH 23152.]\\n- 목차: CHAPTER 1. ...\\n- 내용: It is unlawful..."
    """
    if not docs:
        return "(관련 법률 정보 없음)"

    formatted = []
    for doc in docs:
        meta = doc.metadata
        formatted.append(
            f"[{meta['law_type']} {meta['article_no']}]\n"
            f"- 목차: {meta['section_title']}\n"
            f"- 내용: {doc.page_content}\n"
            f"--------------------------------------------------"
        )
    return "\n".join(formatted)


def _extract_text_from_ai_content(content: Any) -> str:
    """AIMessage content를 안전하게 문자열로 변환합니다."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content) if content is not None else ""


# =========================================================
# 7. [메인] RAG 답변 생성 함수 — API 엔드포인트의 진입점
# =========================================================


@traceable
async def generate_answer(
    query: str, db: AsyncSession, country_id: int, session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    사용자의 법률 질문에 대해 RAG 기반 답변을 생성합니다.

    [전체 흐름]
    번역 → 검색 → 포맷 → LCEL 체인(프롬프트 → LLM → 파싱) → 응답

    Args:
        query: 사용자의 원본 질문 (한국어 또는 영어)
        db: 비동기 DB 세션
        country_id: 검색 대상 국가 ID

    Returns:
        {
            "answer": "AI가 생성한 답변 텍스트",
            "related_law_id_list": [68722, 68724],  # 참고한 법률 ID
            "search_success": True/False  # 관련 법률을 찾았는지 여부
        }
    """
    logger.info("국가 필터링 적용: ID %s", country_id)
    # Step 0: 질문 재구성 — session_id가 있으면 대화 맥락을 반영해
    # 후속 질문을 독립 질문으로 재구성 (예: "그러면 벌금은?" → "캘리포니아 음주운전 벌금은?")
    search_query = query
    if session_id:
        search_query = await contextualize_question(query, session_id, llm)

    # Step 1: 질문 번역 (한국어 → 영어, 검색용. 원본 질문은 답변 생성에 사용)
    translated_query = await translate_query(search_query)

    # Step 2: 벡터 검색
    docs, law_ids = await retrieve_laws(translated_query, country_id, db)

    # Step 3: 유효한 문서가 없으면 LLM 호출 없이 바로 "검색 실패" 응답 (비용/시간 절약)
    if not docs:
        return {
            "answer": "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다. (관련도 낮음)",
            "related_law_id_list": [],
            "search_success": False,
        }

    # Step 4: 컨텍스트 포맷
    context = format_docs(docs)

    # Step 5: 답변 생성 — 번역본이 아닌 원본 질문(query)을 전달해 답변 언어를 맞춘다
    chain = CHAT_PROMPT | llm

    ai_response = await chain.ainvoke({"context": context, "question": query})
    finish_reason = None
    if hasattr(ai_response, "response_metadata") and isinstance(
        ai_response.response_metadata, dict
    ):
        finish_reason = ai_response.response_metadata.get("finish_reason")

    # 디버깅용 로그: 응답 본문에는 노출하지 않습니다.
    logger.info("[qna] finish_reason=%s", finish_reason)

    final_answer = _extract_text_from_ai_content(ai_response.content)

    # Step 6: 대화 기록 저장 (다음 요청의 질문 재구성에 사용)
    if session_id:
        save_to_history(session_id, search_query, final_answer)

    return {
        "answer": final_answer,
        "related_law_id_list": law_ids,
        "search_success": True,
    }
