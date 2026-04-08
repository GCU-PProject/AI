# src/scripts/risk_generate_cards.py
"""
리스크 카드 배치 생성 스크립트

2단계 LLM 파이프라인으로 리스크 카드를 생성하고 JSON 파일로 저장합니다.
DB 적재는 db_load_risk_cards.py에서 별도로 수행합니다.

[전체 흐름]
1. 입력 조합 생성 (국가 × 체류목적 × 비자유형 × 연령대)
2. STEP 1: LLM으로 리스크 주제 6개 생성
3. STEP 2: 각 주제를 DB 법률로 검색 → 법률 있으면 본문 생성, 없으면 제외
4. 검증 + data/risk_cards.json 저장

[실행 방법]
python -m src.scripts.risk_generate_cards                    # 전체 실행
python -m src.scripts.risk_generate_cards --country-id 1     # 특정 국가만
python -m src.scripts.risk_generate_cards --dry-run           # STEP 1만 확인
"""

import asyncio
import argparse
import itertools
import json
import os
import sys
from datetime import datetime

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.output_parsers import JsonOutputParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from src.core.config import settings
from src.core.models import Country
from src.services.chat_service import retrieve_laws, format_docs


# =========================================================
# 1. 설정
# =========================================================
# 입력 조합 enum 값들
# 전체 조합 = 국가 수 × 5(목적) × 4(비자) × 5(연령)
# 예: 2개 국가 기준 → 2 × 5 × 4 × 5 = 200 조합
TRAVEL_PURPOSES = ["tourism", "business", "study", "work", "working_holiday"]
VISA_TYPES = ["short_stay", "long_stay", "work_permit", "student_visa"]
AGE_BANDS = ["10s", "20s", "30s", "40s", "50s_plus"]

# 검증용 상수
VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}
LEVEL_PRIORITY = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# 출력 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "risk_cards.json")


# =========================================================
# 2. LLM + 프롬프트 초기화
# =========================================================
# temperature=0: 일관된 법률 분석 결과 보장
# max_output_tokens=4096: 10개 주제 JSON 배열을 충분히 담을 수 있는 크기
llm = ChatVertexAI(
    model_name="gemini-2.5-pro",  # 데이터 구축의 최상위 품질을 위해 2.5 Pro 모델 강제 고정
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    temperature=0,
    max_output_tokens=8192,
)

parser = JsonOutputParser()

# STEP 1 프롬프트: 리스크 주제 생성 (퓨샷 예시 포함)
topics_yaml = load_prompt("src/prompts/risk_card_topics.yaml", encoding="utf-8")
TOPICS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", topics_yaml.template),
        ("human", "위 조건에 맞는 법적 리스크 주제 6개를 JSON으로 생성해주세요."),
    ]
)

# STEP 2 프롬프트: 법률 기반 카드 본문 생성 (퓨샷 예시 포함)
content_yaml = load_prompt("src/prompts/risk_card_content.yaml", encoding="utf-8")
CONTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", content_yaml.template),
        ("human", "위 법률 조항을 기반으로 리스크 카드 본문을 JSON으로 작성해주세요."),
    ]
)


# =========================================================
# 3. STEP 1: 카드 주제 생성
# =========================================================
# LLM의 사전 학습 지식을 활용하여 해당 국가에서 특히 위험한 법적 리스크를 뽑습니다.
# 10개를 생성하여, STEP 2에서 법령 미매칭으로 탈락될 여분을 확보합니다.
async def generate_topics(
    country_name: str,
    travel_purpose: str,
    visa_type: str,
    age_band: str,
) -> list[dict]:
    """
    LLM으로 리스크 주제 목록 생성.

    Returns:
        [{"risk_title": str, "risk_level": str, "search_query": str}, ...]
        실패 시 빈 리스트 반환.
    """
    chain = TOPICS_PROMPT | llm | parser
    try:
        topics = await chain.ainvoke(
            {
                "country_name": country_name,
                "travel_purpose": travel_purpose,
                "visa_type": visa_type,
                "age_band": age_band,
            }
        )
        if not isinstance(topics, list):
            print(f"  ⚠️ LLM 응답이 리스트가 아닙니다: {type(topics)}")
            return []
        return topics
    except Exception as e:
        print(f"  ❌ STEP 1 실패: {e}")
        return []


# =========================================================
# 4. STEP 2: 법령 검색 + 카드 본문 생성
# =========================================================
# STEP 1에서 생성된 각 주제에 대해:
# 1) search_query로 DB 법률을 벡터 검색 (retrieve_laws 재사용)
# 2) 법률이 있으면 → LLM으로 카드 본문(risk_content, risk_actions) 생성
# 3) 법률이 없으면 → 이 카드 제외 (근거 없는 카드 비공개 원칙)
async def generate_card_with_law(
    topic: dict,
    country_id: int,
    travel_purpose: str,
    visa_type: str,
    age_band: str,
    db: AsyncSession,
) -> dict | None:
    """
    주제 1개에 대해 DB 법률 검색 → 본문 생성.

    Returns:
        완성된 카드 dict. 법률 미발견 시 None.
    """
    search_query = topic.get("search_query", topic.get("risk_title", ""))

    # DB 법률 검색 (chat_service.py의 retrieve_laws 재사용)
    docs, law_ids = await retrieve_laws(search_query, country_id, db)

    # 사용자가 "관련법령 1개만 출력"을 요청하여, 가장 유사도 높은 상위 1개로 제한해 토큰(비용) 절약
    docs = docs[:1]

    if not docs:
        return None

    # 검색된 법률 기반으로 카드 본문 생성
    context = format_docs(docs)
    chain = CONTENT_PROMPT | llm | parser
    try:
        result = await chain.ainvoke(
            {
                "risk_title": topic["risk_title"],
                "risk_level": topic["risk_level"],
                "context": context,
                "travel_purpose": travel_purpose,
                "visa_type": visa_type,
                "age_band": age_band,
            }
        )
    except Exception as e:
        print(f"    ❌ STEP 2 LLM 실패 ({topic['risk_title']}): {e}")
        return None

    # 카드 조립 (law_refs는 실제 검색된 법률 정보로 매핑)
    return {
        "risk_title": topic["risk_title"],
        "risk_level": topic.get("risk_level", "MEDIUM"),
        "risk_content": result.get("risk_content", ""),
        "risk_actions": result.get("risk_actions", []),
        "law_refs": [
            {
                "law_id": d.metadata["law_id"],
                "law_type": d.metadata["law_type"],
                "article_no": d.metadata["article_no"],
            }
            for d in docs
        ],
        "issue_refs": [],  # 현재 이슈 기능 미구현 → 빈 배열
    }


# =========================================================
# 5. 검증
# =========================================================
def validate_cards(cards: list[dict]) -> list[dict]:
    """
    카드 목록을 검증하고 정제합니다.
    1. 필수 필드 누락 제거 (risk_title, risk_level, risk_content, risk_actions)
    2. risk_level 값 검증 (HIGH/MEDIUM/LOW만 허용)
    3. 중복 제거 (risk_title 기준)
    4. law_refs 빈 배열 제거 (안전장치)
    """
    valid, seen = [], set()
    for card in cards:
        if not all(
            card.get(f)
            for f in ["risk_title", "risk_level", "risk_content", "risk_actions"]
        ):
            continue
        if card["risk_level"] not in VALID_LEVELS:
            card["risk_level"] = "MEDIUM"
        if card["risk_title"] in seen or not card.get("law_refs"):
            continue
        seen.add(card["risk_title"])
        valid.append(card)
    return valid


def calculate_overall_level(cards: list[dict]) -> str:
    """카드 중 가장 높은 위험 등급을 overall_risk_level로 산정."""
    if not cards:
        return "LOW"
    max_p = max(LEVEL_PRIORITY.get(c["risk_level"], 1) for c in cards)
    return {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(max_p, "MEDIUM")


# =========================================================
# 6. 국가 정보 조회
# =========================================================
async def get_all_countries(db: AsyncSession) -> list[dict]:
    """DB의 countries 테이블에서 전체 국가 목록을 조회합니다."""
    result = await db.execute(select(Country))
    return [
        {
            "country_id": c.country_id,
            "country_name": (
                f"{c.country_name} ({c.state_name})" if c.state_name else c.country_name
            ),
        }
        for c in result.scalars().all()
    ]


# =========================================================
# 7. 메인 실행
# =========================================================
async def main(country_id_filter: int | None = None, dry_run: bool = False):
    print("🚀 리스크 카드 배치 생성 시작")
    print(f"   모드: {'드라이런 (STEP 1만)' if dry_run else '전체 실행'}")

    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    all_results = []

    async with async_session() as db:
        # 국가 목록 조회 (필터 적용)
        countries = await get_all_countries(db)
        if country_id_filter:
            countries = [c for c in countries if c["country_id"] == country_id_filter]
        if not countries:
            print("❌ 대상 국가가 없습니다.")
            return

        print(f"📋 대상 국가: {[c['country_name'] for c in countries]}")

        # 전체 조합 생성 (국가 × 목적 × 비자 × 연령)
        combinations = list(
            itertools.product(countries, TRAVEL_PURPOSES, VISA_TYPES, AGE_BANDS)
        )
        total = len(combinations)
        print(f"📊 총 조합 수: {total}")

        for idx, (country, purpose, visa, age) in enumerate(combinations, 1):
            cid, cname = country["country_id"], country["country_name"]
            print(f"\n[{idx}/{total}] 🌍 {cname} | {purpose} | {visa} | {age}")

            # ─── STEP 1: 주제 생성 ───
            print("  📝 STEP 1: 주제 생성 중...")
            topics = await generate_topics(cname, purpose, visa, age)
            if not topics:
                print("  ⚠️ 주제 생성 실패, 건너뜁니다.")
                continue
            print(f"  ✅ 주제 {len(topics)}개 생성됨")

            # 드라이런: 주제만 출력하고 STEP 2는 건너뜀
            if dry_run:
                for t in topics:
                    print(
                        f"    - [{t.get('risk_level', '?')}] {t.get('risk_title', '?')}"
                    )
                all_results.append(
                    {
                        "country_id": cid,
                        "travel_purpose": purpose,
                        "visa_type": visa,
                        "age_band": age,
                        "overall_risk_level": "PENDING",
                        "risk_list": [],
                        "_meta": {
                            "generated_at": datetime.now().isoformat(),
                            "mode": "dry_run",
                            "total_topics": len(topics),
                            "topics_preview": [t.get("risk_title", "") for t in topics],
                        },
                    }
                )
                continue

            # ─── STEP 2: 법령 검색 + 본문 생성 ───
            print("  🔍 STEP 2: 법령 검색 + 본문 생성 중...")
            cards, filtered_reasons = [], {}

            for t_idx, topic in enumerate(topics, 1):
                title = topic.get("risk_title", f"주제{t_idx}")
                print(f"    [{t_idx}/{len(topics)}] {title}...", end=" ")

                card = await generate_card_with_law(topic, cid, purpose, visa, age, db)
                if card:
                    print("✅")
                    cards.append(card)
                else:
                    filtered_reasons["법령 미발견"] = (
                        filtered_reasons.get("법령 미발견", 0) + 1
                    )
                    print("❌ (법령 미발견)")

            # ─── 검증 + sort_order 부여 ───
            valid_cards = validate_cards(cards)
            if len(cards) - len(valid_cards) > 0:
                filtered_reasons["검증 실패"] = len(cards) - len(valid_cards)

            for i, card in enumerate(valid_cards, 1):
                card["sort_order"] = i

            overall = calculate_overall_level(valid_cards)
            filtered_total = len(topics) - len(valid_cards)

            print(
                f"  📊 결과: {len(topics)}개 주제 → {len(valid_cards)}개 카드 (제거: {filtered_total}개)"
            )
            if len(valid_cards) < 3:
                print("  ⚠️ 경고: 최종 카드 3개 미만. DB 법률 데이터 보강 필요.")

            all_results.append(
                {
                    "country_id": cid,
                    "travel_purpose": purpose,
                    "visa_type": visa,
                    "age_band": age,
                    "overall_risk_level": overall,
                    "risk_list": valid_cards,
                    "_meta": {
                        "generated_at": datetime.now().isoformat(),
                        "total_topics": len(topics),
                        "filtered_count": filtered_total,
                        "filter_reasons": filtered_reasons,
                    },
                }
            )

    # ─── JSON 저장 ───
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_cards = sum(len(r["risk_list"]) for r in all_results)
    print(f"\n💾 저장 완료: {OUTPUT_FILE}")
    print(f"📊 총 {len(all_results)}개 조합, {total_cards}개 카드")
    if not dry_run:
        total_filtered = sum(
            r.get("_meta", {}).get("filtered_count", 0) for r in all_results
        )
        print(f"   제거된 카드: {total_filtered}개")

    await engine.dispose()
    print("🎉 배치 생성 완료!")


# =========================================================
# 8. CLI
# =========================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="리스크 카드 배치 생성")
    p.add_argument("--country-id", type=int, default=None, help="특정 국가만 실행")
    p.add_argument("--dry-run", action="store_true", help="STEP 1만 실행")
    args = p.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(country_id_filter=args.country_id, dry_run=args.dry_run))
