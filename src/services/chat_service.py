# src/services/chat_service.py
"""
LangChain 기반 법률 Q&A 서비스 (RAG - Retrieval-Augmented Generation)

이 파일은 사용자의 법률 질문에 대해 관련 법률 조항을 검색(Retrieval)하고,
검색된 조항을 근거로 AI가 답변을 생성(Generation)하는 RAG 파이프라인을 구현합니다.

LangChain 프레임워크를 사용하여 각 단계를 독립적인 컴포넌트로 분리하고,
LCEL(LangChain Expression Language) 파이프라인으로 연결합니다.

[전체 처리 흐름]
사용자 질문 (한국어)
    → 1단계: 질문을 영어로 번역 (검색 정확도 향상)
    → 2단계: 번역된 질문을 벡터(숫자 배열)로 변환 (임베딩)
    → 3단계: DB에서 벡터 유사도 기반으로 관련 법률 조항 검색
    → 4단계: 검색 결과를 프롬프트에 삽입
    → 5단계: LLM(Gemini)이 근거 자료 기반으로 답변 생성
    → 6단계: API 응답 반환
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.llm import embeddings, get_llm
from src.models import Country, Law
from src.services.memory import contextualize_question, save_to_history

logger = logging.getLogger(__name__)

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
# 이 값들은 RAG 검색 품질에 직접 영향을 미치며,
# 데이터 특성이나 사용 환경에 따라 조정이 필요할 수 있습니다.

# TOP_K: 벡터 검색에서 가져올 최대 문서 수
# - 사용자의 질문과 가장 유사한 법률 조항을 상위 몇 개까지 가져올지 결정합니다.
# - 너무 작으면(예: 2~3) 관련 법률을 놓칠 수 있고,
#   너무 크면(예: 20) 관련 없는 문서가 섞여 답변 품질이 저하됩니다.
# - 검색 정확도(Recall)를 극대화하고 누락을 방지하기 위해 7로 설정했습니다.
TOP_K = settings.RAG_TOP_K

# MAX_DISTANCE_THRESHOLD: L2 거리(유클리드 거리) 기반 유사도 임계값
# - 벡터 간 거리가 이 값 이하인 문서만 "관련 있음"으로 판단합니다.
# - 거리가 0에 가까울수록 질문과 문서가 의미적으로 유사합니다.
# - 이 값보다 거리가 큰 문서는 관련성이 낮다고 판단하여 제외합니다.
# - 예: 질문 "음주운전 처벌"과 문서 "주차 위반"의 거리가 0.92라면,
#        0.92 > 0.85 이므로 이 문서는 검색 결과에서 제외됩니다.
MAX_DISTANCE_THRESHOLD = settings.RAG_MAX_DISTANCE_THRESHOLD

# =========================================================
# 2. AI 모델 초기화
# =========================================================
# 이 모듈이 import될 때 한 번만 실행됩니다.
# LangChain에서는 객체를 모듈 로드 시 한 번만 생성하면
# 이후 모든 요청에서 재사용됩니다.

llm = get_llm()

# =========================================================
# 3. 질문 번역 기능 (검색 정확도 향상을 위한 전처리)
# =========================================================
# [문제] DB에 저장된 법률 데이터는 영어인데, 사용자가 한국어로 질문하면
#        한국어 벡터와 영어 벡터 간의 거리가 커져서 검색 정확도가 떨어집니다.
#        예: "음주운전 처벌" (한국어 벡터) ↔ "DUI penalties" (영어 벡터) → 거리가 큼
#
# [해결] 검색 전에 질문을 영어로 번역하여 임베딩합니다.
#        번역된 질문은 검색에만 사용하고, 최종 답변 생성 시에는
#        원본 질문(한국어)을 그대로 사용하여 한국어로 답변합니다.
#
# [흐름]
# 한국어 질문 → [영어 번역] → 임베딩 → 벡터 검색 (여기까지 번역본 사용)
#                                                  ↓
#               원본 한국어 질문 + 검색 결과 → LLM → 한국어 답변 생성
#
# [비용] 별도 번역 API 없이 기존 LLM(Gemini)을 재사용하므로 추가 비용 없음

# 번역용 프롬프트 템플릿
# - ChatPromptTemplate: LangChain에서 프롬프트를 관리하는 클래스
# - from_messages(): 대화 형식(system/human)으로 프롬프트를 구성
# - system 메시지: AI의 역할과 규칙을 정의
# - human 메시지: 사용자의 입력. {query}는 실행 시 실제 질문으로 치환됨

translation_yaml = load_prompt("src/prompts/translation.yaml", encoding="utf-8")

TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", translation_yaml.template),
        ("human", "{query}"),
    ]
)


async def translate_query(query: str) -> str:
    """
    사용자의 질문을 영어로 번역합니다. (벡터 검색 정확도 향상 목적)

    - 이미 영어인 경우 그대로 반환합니다.
    - 번역 결과는 검색에만 사용되며, 최종 답변 생성에는 원본 질문이 사용됩니다.

    LCEL 파이프라인 설명:
        translation_prompt | llm | StrOutputParser()
        (프롬프트 생성)   → (LLM 호출) → (응답에서 텍스트만 추출)
        이 파이프라인은 | (파이프) 연산자로 각 단계를 연결합니다.
        데이터가 왼쪽에서 오른쪽으로 순차적으로 흐릅니다.
    """
    chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    translated_res = await chain.ainvoke({"query": query})
    
    # [중요] 최신 LangChain 버전에서 StrOutputParser()의 출력이 문자열이 아닌
    # TextAccessor 객체로 반환될 수 있습니다. 이를 순수 str 자료형으로 강제 변환해주어야
    # 임베딩 라이브러리(GoogleGenerativeAIEmbeddings)에서 'Empty instances' 에러가 나는 것을 방지할 수 있습니다.
    translated = str(translated_res)
    
    if not translated or not translated.strip():
        raise ValueError("번역 결과가 없습니다.")
    logger.info("번역 완료: '%s' → '%s'", query, translated)
    return translated


# =========================================================
# 4. RAG 프롬프트 템플릿 (답변 생성용)
# =========================================================
# LLM에게 전달할 시스템 프롬프트입니다.
# 이 프롬프트는 LLM의 행동 규칙, 답변 형식, 가드레일(안전장치)을 정의합니다.
#
# [프롬프트 설계 핵심]
# 1. 가드레일 (할루시네이션 방지):
#    - "반드시 근거 자료만 사용" → LLM이 학습 데이터에서 지어내는 것을 방지
#    - "관련 없으면 답변 불가 처리" → 무관한 자료로 억지 답변 생성 방지
# 2. 답변 형식 통일:
#    - 결론 → 상세 내용 → 참고 법령 → 면책 조항 순서로 구조화
# 3. 인용 방식:
#    - 본문에 법률 코드가 섞이면 가독성이 떨어지므로, [참고 법령] 섹션으로 분리

chat_yaml = load_prompt("src/prompts/chat.yaml", encoding="utf-8")

# ChatPromptTemplate: LangChain에서 LLM에 전달할 프롬프트를 구조화하는 클래스
# - ("system", SYSTEM_PROMPT): AI의 역할과 규칙 정의. {context}는 검색된 법률 텍스트로 치환
# - ("human", "{question}"): 사용자의 원본 질문. 한국어 그대로 전달하여 한국어 답변 유도
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", chat_yaml.template),
        ("human", "{question}"),
    ]
)


# =========================================================
# 5. 벡터 검색 함수 (커스텀 Retriever)
# =========================================================
# [왜 커스텀 검색 함수를 만들었는가?]
# LangChain에는 PGVector라는 벡터 스토어가 내장되어 있지만,
# 이것은 LangChain 자체 테이블 구조를 요구합니다.
# 우리는 이미 laws 테이블에 법률 데이터와 임베딩을 저장해두었으므로,
# 기존 테이블 구조를 변경하지 않기 위해 SQLAlchemy 쿼리를 직접 작성합니다.
#
# [검색 결과를 LangChain Document로 변환하는 이유]
# LangChain의 다른 컴포넌트(프롬프트, 체인 등)와 호환되려면
# 검색 결과를 LangChain의 표준 데이터 형식인 Document 객체로 변환해야 합니다.
# Document는 page_content(본문)와 metadata(메타데이터)로 구성됩니다.


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


async def retrieve_laws(
    query: str, country_id: int, db: AsyncSession
) -> tuple[List[Document], List[int]]:
    """
    벡터 유사도 기반으로 관련 법률 조항을 검색합니다.

    [처리 과정]
    1. 질문 텍스트를 768차원 벡터로 변환 (임베딩)
    2. DB의 모든 법률 벡터와 L2 거리(유클리드 거리) 계산
    3. 지정된 국가(country_id)의 법률 중 거리가 가장 가까운 TOP_K개 조회
    4. 임계값(MAX_DISTANCE_THRESHOLD) 이하인 문서만 유효한 결과로 반환

    Args:
        query: 검색할 질문 (영어로 번역된 상태)
        country_id: 검색 대상 국가 ID (예: 1=캘리포니아, 2=뉴욕)
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

    # embed_query()는 텍스트를 1024차원 숫자 배열로 변환합니다. (Qwen3-Embedding-0.6B)
    # 예: "DUI penalties" → [0.012, -0.034, 0.056, ..., 0.078] (1024개)
    try:
        query_vector = embeddings.embed_query(query_str)
    except Exception as e:
        logger.error("❌ embed_query 중 오류 발생! 입력 쿼리: %r, 에러: %s", query_str, e)
        raise


    # Step 2: DB에서 벡터 유사도 검색 (SQLAlchemy + pgvector)
    # - Law.embedding.l2_distance(query_vector): 질문 벡터와 각 법률 벡터 간의 L2 거리 계산
    #   L2 거리 = 두 벡터 간의 유클리드 거리. 값이 작을수록 의미적으로 유사
    # - .where(country_id == ...): 특정 국가의 법률만 필터링
    # - .order_by(distance): 거리가 가까운(유사한) 순서로 정렬
    # - .limit(TOP_K): 상위 5개만 가져옴
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
        law = row[0]  # Law 객체 (법률 데이터)
        distance = row[1]  # L2 거리 값 (0에 가까울수록 유사)

        # Step 3: 임계값 필터링
        # TOP_K개를 가져왔더라도, 거리가 임계값(0.85)보다 크면
        # 관련성이 낮다고 판단하여 제외합니다.
        # 예: 거리 0.72 → 유효 (0.72 ≤ 0.85) ✅
        #     거리 0.91 → 제외 (0.91 > 0.85) ❌
        if distance <= MAX_DISTANCE_THRESHOLD:
            # Step 4: LangChain Document 형식으로 변환
            # - page_content: 법률 본문 텍스트 (프롬프트에 삽입될 내용)
            # - metadata: 부가 정보 (법률 종류, 조항 번호, 거리 등)
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
# 검색된 법률 Document들을 LLM 프롬프트에 삽입할 수 있는
# 문자열(텍스트) 형태로 변환합니다.
#
# [포맷 예시]
# [VEH 23152.]
# - 목차: CHAPTER 1. Offenses and Penalties
# - 내용: It is unlawful for a person who is under the influence...
# --------------------------------------------------
# [VEH 23153.]
# - 목차: CHAPTER 1. Offenses and Penalties
# - 내용: ...


def format_docs(docs: List[Document]) -> str:
    """
    LangChain Document 리스트를 프롬프트에 삽입할 텍스트로 변환합니다.

    각 문서를 [법률코드 조항번호] 헤더와 함께 목차, 내용을 포함하는
    구조화된 텍스트로 포맷합니다. 구분선으로 문서 간 경계를 명확히 합니다.
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
# 7. [메인] RAG 답변 생성 함수
# =========================================================
# 이 함수가 API 엔드포인트에서 호출되는 최종 진입점입니다.
# 위에서 정의한 모든 컴포넌트(번역, 검색, 포맷, 프롬프트, LLM)를
# 순차적으로 조합하여 최종 답변을 생성합니다.


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
    # ----- Step 0: 질문 재구성 (대화 맥락 반영) -----
    # session_id가 있으면 이전 대화 기록을 참고하여 후속 질문을 독립적 질문으로 재구성
    # 예: "그러면 벌금은?" → "캘리포니아 음주운전 DUI 벌금은?"
    # session_id가 없으면 (None) 원본 질문을 그대로 사용
    search_query = query
    if session_id:
        search_query = await contextualize_question(query, session_id, llm)

    # ----- Step 1: 질문 번역 (한국어 → 영어) -----
    # 검색 정확도를 위해 질문을 영어로 번역합니다.
    # 재구성된 질문(search_query)을 번역합니다.
    # 원본 질문(query)은 보존하고, 최종 답변 생성에서 사용합니다.
    translated_query = await translate_query(search_query)

    # ----- Step 2: 벡터 검색 (번역된 질문으로) -----
    # 영어로 번역된 질문을 임베딩하여 DB에서 유사한 법률 조항을 검색합니다.
    docs, law_ids = await retrieve_laws(translated_query, country_id, db)

    # ----- Step 3: 검색 결과 검증 -----
    # 임계값(0.85)을 통과한 유효한 문서가 없으면
    # LLM을 호출하지 않고 바로 "검색 실패" 응답을 반환합니다.
    # (불필요한 LLM 호출을 방지하여 비용과 응답 시간 절약)
    if not docs:
        return {
            "answer": "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다. (관련도 낮음)",
            "related_law_id_list": [],
            "search_success": False,
        }

    # ----- Step 4: 컨텍스트 포맷 -----
    # 검색된 Document 리스트를 프롬프트에 삽입할 텍스트로 변환합니다.
    context = format_docs(docs)

    # ----- Step 5: LCEL 체인 실행 (답변 생성) -----
    # LCEL(LangChain Expression Language) 파이프라인:
    #   prompt | llm
    #   (프롬프트 생성) → (LLM 호출)
    #
    # ※ 여기서는 번역본이 아닌 원본 질문(query)을 전달합니다.
    #   이유: 사용자가 한국어로 질문했으면 한국어로 답변해야 하므로,
    #         LLM에게는 원본 한국어 질문을 전달하여 답변 언어를 맞춥니다.
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

    # ----- Step 6: 대화 기록 저장 -----
    # session_id가 있으면 이번 질문/답변을 대화 기록에 추가
    # 다음 요청에서 contextualize_question()이 이 기록을 참고합니다.
    if session_id:
        save_to_history(session_id, search_query, final_answer)

    # ----- Step 7: 결과 반환 -----
    return {
        "answer": final_answer,
        "related_law_id_list": law_ids,
        "search_success": True,
    }
