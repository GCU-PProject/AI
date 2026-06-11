# src/services/compare_service.py
"""
LangChain 기반 법률 비교 서비스

하나의 질문으로 두 국가의 법률을 각각 검색한 뒤(Double Retrieval),
LLM이 공통점/차이점을 분석해 JSON으로 반환한다.

검색·번역·포맷은 chat_service.py의 함수를 재사용하며,
차이점은 검색이 2회이고 출력이 JsonOutputParser로 파싱된 JSON이라는 것.

[처리 흐름]
국가 조회 → 번역 → 검색(×2) → 검증 → 포맷 → LLM 비교 분석(JSON) → 응답
"""

import logging
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.llm import get_llm
from src.models import Country
from src.services.chat_service import (
    format_docs,  # Document 리스트 → 프롬프트 텍스트 변환 함수
    retrieve_laws,  # 벡터 유사도 기반 법률 검색 함수
    translate_query,  # 한국어 → 영어 번역 함수
)

logger = logging.getLogger(__name__)

# =========================================================
# 1. AI 모델 초기화 (모듈 로드 시 1회 생성, 전 요청 재사용)
# =========================================================
llm = get_llm()

# =========================================================
# 2. 비교 분석용 프롬프트 템플릿
# =========================================================
# [chat_service의 프롬프트와의 차이점]
# - chat_service: {context} 1개 (단일 국가) → 자유 텍스트 답변
# - compare_service: {context_1}, {context_2} 2개 (두 국가) → JSON 형식 답변

# JsonOutputParser: LLM의 응답을 자동으로 JSON(Python dict)으로 변환
# 마크다운 코드 블록(```json ... ```) 안에 있어도 정상적으로 파싱합니다.
parser = JsonOutputParser()

# 비교 분석용 프롬프트 템플릿
# - ("system", COMPARE_SYSTEM_PROMPT): AI 역할, 근거 자료, 가이드라인 정의
#   {context_1}, {context_2}, {format_instructions}는 실행 시 치환됨
# - ("human", "{question}"): 사용자의 원본 질문 (한국어 그대로)

compare_yaml = load_prompt("src/prompts/compare.yaml", encoding="utf-8")

COMPARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", compare_yaml.template),
        ("human", "{question}"),
    ]
)


# =========================================================
# 3. [메인] 법률 비교 서비스
# =========================================================
# 이 함수가 API 엔드포인트(/api/compare)에서 호출되는 최종 진입점입니다.


@traceable
async def compare_laws(
    query: str, db: AsyncSession, country_id_1: int, country_id_2: int
) -> Dict[str, Any]:
    """
    두 국가의 법률을 비교 분석합니다.

    [전체 흐름]
    국가 조회 → 번역 → 검색(×2) → 검증 → 포맷 → LCEL 체인(프롬프트 → LLM → JSON 파싱) → 응답

    Args:
        query: 사용자의 원본 질문 (한국어 또는 영어)
        db: 비동기 DB 세션
        country_id_1: 기준 국가 ID (예: 2=캘리포니아)
        country_id_2: 비교 국가 ID (예: 3=뉴욕)

    Returns:
        {
            "search_success": True/False,
            "country_1_result": {
                "related_law_ids": [68722, 68724],
                "summary": "기준 국가 법률 요약..."
            },
            "country_2_result": {
                "related_law_ids": [12345],
                "summary": "비교 국가 법률 요약..."
            },
            "compare_summary": {
                "common": "공통점 분석...",
                "diff": "차이점 분석..."
            }
        }
    """

    # Step 1: 국가 정보 조회 — 에러 메시지용 {country_id: country_name} 매핑 (쿼리 1회)
    country_stmt = select(Country).where(
        Country.country_id.in_([country_id_1, country_id_2])
    )
    country_result = await db.execute(country_stmt)
    countries = country_result.scalars().all()
    country_map = {c.country_id: c.country_name for c in countries}

    # Step 2: 질문 번역 (1회 수행, 두 국가 검색에 동일한 번역본 사용)
    translated_query = await translate_query(query)

    # Step 3: 두 국가 각각 벡터 검색 — chat_service.retrieve_laws() 재사용 (설정 동일)
    docs_1, ids_1 = await retrieve_laws(translated_query, country_id_1, db)
    docs_2, ids_2 = await retrieve_laws(translated_query, country_id_2, db)

    # Step 4: 검색 결과 검증 — 둘 다 없으면 LLM 호출 없이 즉시 반환
    if not docs_1 and not docs_2:
        logger.warning(
            "두 국가 모두 관련 법률 검색 실패: query='%s' (country_id_1=%s, country_id_2=%s)",
            query,
            country_id_1,
            country_id_2,
        )
        return {
            "country_1_result": {"related_law_ids": [], "summary": "자료 없음"},
            "country_2_result": {"related_law_ids": [], "summary": "자료 없음"},
            "compare_summary": {
                "common": "두 국가 모두 관련 법률 데이터를 찾을 수 없습니다.",
                "diff": "",
            },
        }

    if not docs_1:
        country_1_name = country_map.get(country_id_1, str(country_id_1))
        context_1_text = f"{country_1_name}의 관련 법률 데이터를 찾을 수 없습니다."
        logger.info(
            "비교 대상 국가 중 일부 법률 데이터 부재: 국가=%s, 질문='%s'",
            country_1_name,
            query,
        )

    else:
        context_1_text = format_docs(docs_1)

    if not docs_2:
        country_2_name = country_map.get(country_id_2, str(country_id_2))
        context_2_text = f"{country_2_name}의 관련 법률 데이터를 찾을 수 없습니다."
        logger.info(
            "비교 대상 국가 중 일부 법률 데이터 부재: 국가=%s, 질문='%s'",
            country_2_name,
            query,
        )

    else:
        context_2_text = format_docs(docs_2)

    # Step 5: 비교 분석 — 프롬프트 → LLM → JSON 파싱
    chain = COMPARE_PROMPT | llm | parser

    # LLM이 출력할 JSON 구조 예시 (프롬프트의 {format_instructions}에 삽입)
    format_instructions = """{
    "summary_1": "기준 국가 법률 요약",
    "summary_2": "비교 국가 법률 요약",
    "common": "공통점 분석",
    "diff": "차이점 분석"
}"""

    # LLM이 JSON 형식을 지키지 않아 파싱에 실패하면 기본값("분석 실패")으로 대체
    try:
        analysis = await chain.ainvoke(
            {
                "context_1": context_1_text,
                "context_2": context_2_text,
                "question": query,
                "format_instructions": format_instructions,
            }
        )
    except Exception as e:
        logger.error("❌ Gemini 호출/파싱 실패: '%s'", e)
        analysis = {
            "summary_1": "분석 실패",
            "summary_2": "분석 실패",
            "common": "오류 발생",
            "diff": "오류 발생",
        }

    # Step 6: 결과 반환 — .get(key, "")로 LLM의 키 누락에 대비
    return {
        "country_1_result": {
            "related_law_ids": ids_1,
            "summary": analysis.get("summary_1", ""),
        },
        "country_2_result": {
            "related_law_ids": ids_2,
            "summary": analysis.get("summary_2", ""),
        },
        "compare_summary": {
            "common": analysis.get("common", ""),
            "diff": analysis.get("diff", ""),
        },
    }
