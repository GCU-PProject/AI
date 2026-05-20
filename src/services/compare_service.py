# src/services/compare_service.py
"""
LangChain 기반 법률 비교 서비스

이 파일은 두 국가의 법률을 비교 분석하는 서비스를 구현합니다.
사용자가 하나의 질문(예: "음주운전 처벌")을 입력하면,
두 국가 각각에서 관련 법률을 검색한 뒤, LLM이 공통점과 차이점을 분석하여
JSON 형식의 비교 결과를 반환합니다.

[chat_service.py와의 관계]
이 파일은 chat_service.py의 핵심 함수들(검색, 포맷, 번역)을 재사용합니다.
중복 코드를 줄이고, 검색/번역 로직이 변경되면 한 곳만 수정하면 됩니다.

| 가져오는 것        | 용도                              |
|--------------------|-----------------------------------|
| retrieve_laws()    | 벡터 유사도 기반 법률 검색         |
| format_docs()      | Document → 프롬프트 텍스트 변환    |
| translate_query()  | 한국어 질문 → 영어 번역 (검색용)   |
| TOP_K              | 검색 시 가져올 최대 문서 수        |
| MAX_DISTANCE_THRESHOLD | 유사도 임계값                  |

[chat_service.py와의 차이점]
| 구분         | chat_service (Q&A)     | compare_service (비교)      |
|--------------|------------------------|-----------------------------|
| 검색 횟수    | 1회 (1개 국가)          | 2회 (2개 국가 각각)          |
| 출력 형식    | 자유 텍스트 (문자열)    | 구조화된 JSON               |
| 출력 파서    | StrOutputParser        | JsonOutputParser             |
| max_tokens   | 4096                   | 2048 (비교 분석이 더 길므로) |

[전체 처리 흐름]
사용자 질문 (한국어)
    → 1단계: 국가 정보 조회 (country_id → country_name 변환)
    → 2단계: 질문을 영어로 번역 (검색 정확도 향상)
    → 3단계: 두 국가 각각 벡터 검색 (Double Retrieval)
    → 4단계: 검색 결과가 없는 국가가 있으면 에러 반환
    → 5단계: 검색 결과를 프롬프트 텍스트로 포맷
    → 6단계: LCEL 체인 실행 (프롬프트 → LLM → JSON 파싱)
    → 7단계: API 응답 형식으로 조립하여 반환
"""

import json
import logging
from typing import Dict, Any
from src.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import load_prompt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.core.models import Country
from src.core.config import settings

logger = logging.getLogger(__name__)

# =========================================================
# chat_service에서 공통 함수/설정 가져오기
# =========================================================
# 검색, 포맷, 번역 로직은 chat_service.py에 이미 구현되어 있으므로
# import하여 재사용합니다. 이렇게 하면:
# 1. 코드 중복을 방지 (DRY 원칙: Don't Repeat Yourself)
# 2. 검색/번역 로직 수정 시 chat_service.py만 변경하면 됨
from src.services.chat_service import (
    retrieve_laws,  # 벡터 유사도 기반 법률 검색 함수
    format_docs,  # Document 리스트 → 프롬프트 텍스트 변환 함수
    translate_query,  # 한국어 → 영어 번역 함수
    TOP_K,  # 검색 시 가져올 최대 문서 수 (5)
    MAX_DISTANCE_THRESHOLD,  # L2 거리 임계값 (0.85)
)

# =========================================================
# 2. AI 모델 초기화
# =========================================================
# 이 모듈이 import될 때 한 번만 실행됩니다.
# LangChain에서는 객체를 모듈 로드 시 한 번만 생성하면
# 이후 모든 요청에서 재사용됩니다.

llm = get_llm()

# =========================================================
# 2. 비교 분석용 프롬프트 템플릿
# =========================================================
# [chat_service의 프롬프트와의 차이점]
# - chat_service: {context} 1개 (단일 국가) → 자유 텍스트 답변
# - compare_service: {context_1}, {context_2} 2개 (두 국가) → JSON 형식 답변
#
# ChatPromptTemplate이 {variable}만 변수로 치환하므로
# 일반 { }를 그대로 사용할 수 있습니다.

# 비교 분석용 시스템 프롬프트
# - {context_1}: 기준 국가의 검색된 법률 텍스트 (format_docs로 변환된 문자열)
# - {context_2}: 비교 국가의 검색된 법률 텍스트
# - {format_instructions}: LLM이 출력해야 할 JSON 형식 예시
#
# [프롬프트 구조]
# NO_DATA_MSG를 문자열 결합(+)으로 프롬프트에 삽입하고 있습니다.
# 이는 ChatPromptTemplate의 변수 치환({variable})과는 다른 방식으로,
# 프롬프트 정의 시점에 이미 값이 고정됩니다.

# JsonOutputParser: LLM의 응답을 자동으로 JSON(Python dict)으로 변환
# LLM 응답에서 JSON 부분을 자동으로 추출하고,
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
        country_id_1: 기준 국가 ID (예: 1=캘리포니아)
        country_id_2: 비교 국가 ID (예: 2=뉴욕)

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

    # ----- Step 1: 국가 정보 조회 (ID → 이름 변환) -----
    # country_id는 숫자(1, 2)이므로, 에러 메시지에서 국가 이름을 표시하기 위해
    # DB에서 {country_id: country_name} 매핑을 미리 조회합니다.
    # 예: {1: "United States (California)", 2: "United States (New York)"}
    #
    # .in_() 연산자로 두 국가를 한 번의 쿼리로 동시에 조회합니다.
    # (2번 쿼리하는 것보다 효율적)
    country_stmt = select(Country).where(
        Country.country_id.in_([country_id_1, country_id_2])
    )
    country_result = await db.execute(country_stmt)
    countries = country_result.scalars().all()
    country_map = {c.country_id: c.country_name for c in countries}

    # ----- Step 2: 질문 번역 (한국어 → 영어) -----
    # chat_service의 translate_query()를 재사용합니다.
    # 번역은 1회만 수행하고, 두 국가 검색에 모두 같은 번역 결과를 사용합니다.
    translated_query = await translate_query(query)

    # ----- Step 3: 두 국가 각각 벡터 검색 (Double Retrieval) -----
    # 같은 질문(번역본)으로 두 국가의 법률을 각각 검색합니다.
    # chat_service의 retrieve_laws()를 재사용하므로,
    # TOP_K, MAX_DISTANCE_THRESHOLD, 임베딩 모델 등 모든 설정이 동일합니다.
    #
    # 반환값:
    # - docs_1/docs_2: LangChain Document 리스트 (프롬프트에 삽입할 법률 텍스트)
    # - ids_1/ids_2: 법률 ID 리스트 (API 응답의 related_law_ids에 사용)
    docs_1, ids_1 = await retrieve_laws(translated_query, country_id_1, db)
    docs_2, ids_2 = await retrieve_laws(translated_query, country_id_2, db)

    # ----- Step 4: 검색 결과 검증 -----
    # 두 국가 중 하나라도 유효한 검색 결과가 없으면 비교가 불가능합니다.
    # 이 경우 LLM을 호출하지 않고 바로 에러 응답을 반환합니다.
    #
    # 에러 메시지에는 어떤 국가에서 데이터를 찾지 못했는지 명시합니다.
    # 예: "United States (California)의 관련 법률 데이터를 찾을 수 없습니다."
    if not docs_1 and not docs_2:

        return {
            "search_success": False,
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

    else:
        context_1_text = format_docs(docs_1)

    if not docs_2:
        country_2_name = country_map.get(country_id_2, str(country_id_2))
        context_2_text = f"{country_2_name}의 관련 법률 데이터를 찾을 수 없습니다."

    else:
        context_2_text = format_docs(docs_2)

    # ----- Step 6: LCEL 체인 실행 (비교 분석) -----
    # LCEL 파이프라인:
    #   prompt | llm | parser
    #   (프롬프트 생성) → (LLM 호출) → (응답을 JSON/dict로 파싱)
    #
    # [chat_service와의 차이]
    # - chat_service: prompt | llm | StrOutputParser() → 문자열 반환
    # - compare_service: prompt | llm | parser(JsonOutputParser) → dict 반환
    chain = COMPARE_PROMPT | llm | parser

    # format_instructions: LLM에게 출력해야 할 JSON 구조를 알려주는 예시
    # 이 텍스트가 프롬프트의 {format_instructions} 자리에 삽입됩니다.
    # LLM은 이 형식을 참고하여 동일한 키(summary_1, summary_2, common, diff)를
    # 가진 JSON을 출력합니다.
    format_instructions = """{
    "summary_1": "기준 국가 법률 요약",
    "summary_2": "비교 국가 법률 요약",
    "common": "공통점 분석",
    "diff": "차이점 분석"
}"""

    # 체인 실행 (프롬프트의 4개 변수에 값을 채워 넣고, LLM 호출 후 JSON 파싱)
    # - context_1: 기준 국가의 법률 텍스트
    # - context_2: 비교 국가의 법률 텍스트
    # - question: 원본 질문 (한국어 → 한국어 답변 유도)
    # - format_instructions: JSON 출력 형식 예시
    #
    # try-except로 감싸는 이유:
    # LLM이 가끔 JSON 형식을 지키지 않아 파싱에 실패할 수 있습니다.
    # 이 경우 에러를 발생시키지 않고, 기본값("분석 실패")으로 대체합니다.
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

    # ----- Step 7: 결과 반환 -----
    # LLM의 JSON 응답(analysis)에서 각 필드를 추출하여
    # API 응답 형식에 맞게 재조립합니다.
    #
    # analysis.get("key", ""): 키가 없을 경우 빈 문자열을 기본값으로 사용
    # (LLM이 일부 키를 누락할 가능성에 대한 안전장치)
    return {
        "search_success": True,
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
