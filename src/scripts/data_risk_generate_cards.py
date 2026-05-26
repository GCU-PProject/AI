"""
data_risk_generate_cards.py

리스크 카드 배치 생성 스크립트

실행 예시:
python -m src.scripts.data_risk_generate_cards
python -m src.scripts.data_risk_generate_cards --country-id 1
python -m src.scripts.data_risk_generate_cards --dry-run
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
from src.models import Country
from src.services.chat_service import retrieve_laws, format_docs


# =========================================================
# 1. 설정
# =========================================================
TRAVEL_PURPOSES = ["tourism", "business", "study", "work", "working_holiday"]
VISA_TYPES = ["short_stay", "long_stay", "work_permit", "student_visa"]
AGE_BANDS = ["10s", "20s", "30s", "40s", "50s_plus"]

VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}
LEVEL_PRIORITY = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "risk_cards.json")


# =========================================================
# 2. LLM + 프롬프트 초기화
# =========================================================
llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    temperature=0,
    max_output_tokens=8192,
)

parser = JsonOutputParser()

topics_yaml = load_prompt("src/prompts/risk_card_topics.yaml", encoding="utf-8")
TOPICS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", topics_yaml.template),
        ("human", "위 조건에 맞는 법적 리스크 주제 6개를 JSON으로 생성해주세요."),
    ]
)

content_yaml = load_prompt("src/prompts/risk_card_content.yaml", encoding="utf-8")
CONTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", content_yaml.template),
        ("human", "위 법률 조항을 기반으로 리스크 카드 본문을 JSON으로 작성해주세요."),
    ]
)


async def generate_topics(
    country_name: str, travel_purpose: str, visa_type: str, age_band: str
) -> list[dict]:
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


async def generate_card_with_law(
    topic: dict,
    country_id: int,
    travel_purpose: str,
    visa_type: str,
    age_band: str,
    db: AsyncSession,
) -> dict | None:
    search_query = topic.get("search_query", topic.get("risk_title", ""))
    docs, law_ids = await retrieve_laws(search_query, country_id, db)
    docs = docs[:1]
    if not docs:
        return None
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
        "issue_refs": [],
    }


def validate_cards(cards: list[dict]) -> list[dict]:
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
    if not cards:
        return "LOW"
    max_p = max(LEVEL_PRIORITY.get(c["risk_level"], 1) for c in cards)
    return {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(max_p, "MEDIUM")


async def get_all_countries(db: AsyncSession) -> list[dict]:
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


async def main(country_id_filter: int | None = None, dry_run: bool = False):
    print("🚀 리스크 카드 배치 생성 시작")
    print(f"   모드: {'드라이런 (STEP 1만)' if dry_run else '전체 실행'}")
    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    all_results = []
    async with async_session() as db:
        countries = await get_all_countries(db)
        if country_id_filter:
            countries = [c for c in countries if c["country_id"] == country_id_filter]
        if not countries:
            print("❌ 대상 국가가 없습니다.")
            return
        print(f"📋 대상 국가: {[c['country_name'] for c in countries]}")
        combinations = list(
            itertools.product(countries, TRAVEL_PURPOSES, VISA_TYPES, AGE_BANDS)
        )
        total = len(combinations)
        print(f"📊 총 조합 수: {total}")
        for idx, (country, purpose, visa, age) in enumerate(combinations, 1):
            cid, cname = country["country_id"], country["country_name"]
            print(f"\n[{idx}/{total}] 🌍 {cname} | {purpose} | {visa} | {age}")
            print("  📝 STEP 1: 주제 생성 중...")
            topics = await generate_topics(cname, purpose, visa, age)
            if not topics:
                print("  ⚠️ 주제 생성 실패, 건너뜁니다.")
                continue
            print(f"  ✅ 주제 {len(topics)}개 생성됨")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="리스크 카드 배치 생성")
    p.add_argument("--country-id", type=int, default=None, help="특정 국가만 실행")
    p.add_argument("--dry-run", action="store_true", help="STEP 1만 실행")
    args = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(country_id_filter=args.country_id, dry_run=args.dry_run))
