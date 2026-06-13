"""
data_risk_generate_cards.py

리스크 카드 배치 생성 스크립트

실행 예시:
python -m src.scripts.data_risk_generate_cards
python -m src.scripts.data_risk_generate_cards --limit 3
python -m src.scripts.data_risk_generate_cards --fresh   # 체크포인트 무시하고 처음부터

[체크포인트]
조합 1개가 끝날 때마다 결과를 OUTPUT_FILE에 즉시 저장합니다.
중간에 중단되더라도 다시 실행하면 완료된 조합은 건너뛰고 이어서 진행합니다.
"""

import argparse
import asyncio
import copy
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

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.models import Country
from src.services.chat_service import retrieve_laws


# =========================================================
# 1. 설정
# =========================================================
TRAVEL_PURPOSES = ["tourism", "business", "study", "work", "working_holiday"]
VISA_TYPES = ["short_stay", "long_stay", "work_permit", "student_visa"]
AGE_BANDS = ["10s", "20s", "30s", "40s", "50s_plus"]

VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}
LEVEL_PRIORITY = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# 병렬 처리 설정 — 순차 실행은 조합당 ~70초라 640조합에 10시간+가 걸린다.
# COMBO_CONCURRENCY: 동시에 처리할 조합 수
# GLOBAL_LLM_CONCURRENCY: 전체 동시 LLM 호출 상한 (429 rate limit 발생 시 낮추세요)
COMBO_CONCURRENCY = 6
GLOBAL_LLM_CONCURRENCY = 12

# 연방(state_code 없음) entry는 여행 목적지가 아니므로 카드 생성 대상에서 제외한다.
# 연방법은 사용자가 주(state)를 선택할 때 resolve_jurisdiction_ids가 자동으로
# 함께 검색하므로 별도 생성이 불필요하다. (프론트도 연방 단위 선택은 막아둠)
# → 미국 연방(1), 캐나다 연방(4), 호주 연방(7), 영국(14, 데이터 없음) 제외

# =========================================================
# 모순 조합 별칭
# =========================================================
# 비현실적인 (목적, 비자) / (목적, 연령) 조합은 LLM 생성을 건너뛰고
# 의미가 같은 조합의 카드를 그대로 복사한다.
# 주의: work + short_stay(불법취업), work + student_visa(알바 제한)처럼
# 상충 자체가 경고 가치인 조합은 별칭으로 처리하지 않는다.
VISA_ALIASES = {
    ("tourism", "work_permit"): "long_stay",
    ("tourism", "student_visa"): "long_stay",
    ("business", "student_visa"): "short_stay",
    ("study", "work_permit"): "student_visa",
    ("working_holiday", "long_stay"): "work_permit",
    ("working_holiday", "student_visa"): "work_permit",
}
# 취업/워홀의 10대 조합은 비현실적 (워홀은 만 18~30세) → 20s 카드 재사용
AGE_ALIASES = {
    ("work", "10s"): "20s",
    ("working_holiday", "10s"): "20s",
}


def resolve_canonical(purpose: str, visa: str, age: str) -> tuple[str, str, str]:
    """별칭 조합을 원본(생성 대상) 조합으로 변환합니다. 별칭이 아니면 그대로 반환."""
    visa = VISA_ALIASES.get((purpose, visa), visa)
    age = AGE_ALIASES.get((purpose, age), age)
    return purpose, visa, age

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "risk_cards.json")


# =========================================================
# 2. LLM + 프롬프트 초기화
# =========================================================
# 배치 생성 전용 모델 — 서비스 모델(.env)과 분리.
# 정적 콘텐츠라 레이턴시가 무관하고, 조항 선별(used_laws)·레벨 재평가 등
# 추론 작업이 많아 thinking 모델을 사용한다.
RISK_CARD_MODEL = "gemini-3.5-flash"

llm = ChatGoogleGenerativeAI(
    model=RISK_CARD_MODEL,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
    temperature=0,
    max_tokens=8192,
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


# 전역 동시 LLM 호출 제한 — 병렬 처리 시 rate limit(429)을 방어한다.
_llm_semaphore = asyncio.Semaphore(GLOBAL_LLM_CONCURRENCY)


async def _ainvoke_with_retry(chain, inputs: dict, retries: int = 1):
    """LLM 호출 헬퍼 — 전역 세마포어로 동시 호출 수를 제한하고, 일시적 오류는 재시도한다."""
    for attempt in range(retries + 1):
        try:
            async with _llm_semaphore:
                return await chain.ainvoke(inputs)
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(2)


async def generate_topics(
    country_name: str, travel_purpose: str, visa_type: str, age_band: str
) -> list[dict]:
    chain = TOPICS_PROMPT | llm | parser
    try:
        topics = await _ainvoke_with_retry(
            chain,
            {
                "country_name": country_name,
                "travel_purpose": travel_purpose,
                "visa_type": visa_type,
                "age_band": age_band,
            },
        )
        if not isinstance(topics, list):
            print(f"  ⚠️ LLM 응답이 리스트가 아닙니다: {type(topics)}")
            return []
        # 필수 키가 없는 주제는 STEP 2에서 어차피 실패하므로 미리 걸러낸다
        return [t for t in topics if isinstance(t, dict) and t.get("risk_title")]
    except Exception as e:
        print(f"  ❌ STEP 1 실패: {e}")
        return []


def format_docs_numbered(docs: list[Document]) -> str:
    """검색된 조항을 [번호]를 붙여 포맷합니다.

    content 프롬프트가 실제 근거로 사용한 조항 번호(used_laws)를
    반환하므로, 번호로 원본 문서를 역추적할 수 있어야 합니다.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        formatted.append(
            f"[{i}] {meta['law_type']} {meta['article_no']}\n"
            f"- 목차: {meta['section_title']}\n"
            f"- 내용: {doc.page_content}\n"
            f"--------------------------------------------------"
        )
    return "\n".join(formatted)


async def search_topic_docs(
    topic: dict, country_id: int, db: AsyncSession
) -> tuple[dict, list[Document]] | None:
    """주제로 법령을 검색한다 (DB 접근 — 세션 공유를 위해 순차 호출).

    검색 자체는 빠르므로 순차로 돌리고, 느린 본문 생성(LLM)은 이후 병렬 처리한다.
    """
    search_query = topic.get("search_query", topic.get("risk_title", ""))
    docs, _law_ids = await retrieve_laws(search_query, country_id, db)
    if not docs:
        return None
    return topic, docs


async def generate_card_content(
    topic: dict,
    docs: list[Document],
    travel_purpose: str,
    visa_type: str,
    age_band: str,
) -> dict | None:
    """검색된 조항으로 카드 본문을 생성한다 (LLM 전용 — DB 미사용이라 병렬 호출 가능).

    검색된 조항을 전부 전달하고, 실제 근거로 사용한 조항만 used_laws로 받는다.
    (top-1만 쓰면 관련 조항이 후순위에 있을 때 엉뚱한 법령이 근거로 기록됨)
    """
    context = format_docs_numbered(docs)
    chain = CONTENT_PROMPT | llm | parser
    try:
        result = await _ainvoke_with_retry(
            chain,
            {
                "risk_title": topic["risk_title"],
                "risk_level": topic.get("risk_level", "MEDIUM"),
                "context": context,
                "travel_purpose": travel_purpose,
                "visa_type": visa_type,
                "age_band": age_band,
            },
        )
        if not isinstance(result, dict):
            return None

        used_indexes = result.get("used_laws", [])
        used_docs = [
            docs[i - 1]
            for i in used_indexes
            if isinstance(i, int) and 1 <= i <= len(docs)
        ]
        # 실제 근거로 쓸 수 있는 조항이 없으면 카드를 만들지 않는다 (할루시네이션 방지)
        if not used_docs:
            return None

        risk_level = result.get("risk_level", "")
        if risk_level not in VALID_LEVELS:
            risk_level = topic.get("risk_level", "MEDIUM")

        return {
            "risk_title": topic["risk_title"],
            "risk_level": risk_level,
            "risk_content": result.get("risk_content", ""),
            "risk_actions": result.get("risk_actions", []),
            "law_refs": [
                {
                    "law_id": d.metadata["law_id"],
                    "law_type": d.metadata["law_type"],
                    "article_no": d.metadata["article_no"],
                }
                for d in used_docs
            ],
            "issue_refs": [],
        }
    except Exception as e:
        print(f"    ❌ 본문 생성 실패 ({topic.get('risk_title', '?')}): {e}")
        return None


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
    result = await db.execute(select(Country).order_by(Country.country_id))
    return [
        {
            "country_id": c.country_id,
            "country_name": (
                f"{c.country_name} ({c.state_name})" if c.state_name else c.country_name
            ),
        }
        for c in result.scalars().all()
        if c.state_code is not None
    ]


# =========================================================
# 3. 체크포인트 — 조합 단위 저장/이어하기
# =========================================================


def _combo_key(country_id: int, purpose: str, visa: str, age: str) -> str:
    return f"{country_id}|{purpose}|{visa}|{age}"


def load_checkpoint(fresh: bool) -> list[dict]:
    if fresh or not os.path.exists(OUTPUT_FILE):
        return []
    try:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        print("⚠️ 기존 결과 파일을 읽을 수 없어 처음부터 시작합니다.")
        return []


def save_checkpoint(results: list[dict]) -> None:
    """임시 파일에 쓴 뒤 교체 — 저장 도중 중단돼도 기존 파일이 깨지지 않게 한다."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    tmp_path = OUTPUT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_FILE)


# =========================================================
# 4. 조합 단위 처리
# =========================================================


async def process_combination(
    country_id: int,
    country_name: str,
    purpose: str,
    visa: str,
    age: str,
    async_session,
) -> dict | None:
    # STEP 1: 주제 생성
    topics = await generate_topics(country_name, purpose, visa, age)
    if not topics:
        return None

    # STEP 2: 검색은 한 세션을 공유하므로 순차로(빠름),
    #         본문 생성은 LLM 전용이라 병렬로(느림) 처리한다.
    async with async_session() as db:
        prepared = []
        for topic in topics:
            found = await search_topic_docs(topic, country_id, db)
            if found:
                prepared.append(found)

    cards_raw = await asyncio.gather(
        *[
            generate_card_content(topic, docs, purpose, visa, age)
            for topic, docs in prepared
        ]
    )
    cards = [c for c in cards_raw if c]

    valid_cards = validate_cards(cards)
    for i, card in enumerate(valid_cards, 1):
        card["sort_order"] = i
    overall = calculate_overall_level(valid_cards)
    filtered_total = len(topics) - len(valid_cards)

    return {
        "country_id": country_id,
        "travel_purpose": purpose,
        "visa_type": visa,
        "age_band": age,
        "overall_risk_level": overall,
        "risk_list": valid_cards,
        "_meta": {
            "generated_at": datetime.now().isoformat(),
            "model": RISK_CARD_MODEL,
            "total_topics": len(topics),
            "filtered_count": filtered_total,
        },
    }


# =========================================================
# 5. 메인 루프
# =========================================================


async def main(limit: int | None = None, fresh: bool = False):
    print("🚀 리스크 카드 배치 생성 시작")
    print(f"🤖 사용 모델: {RISK_CARD_MODEL}")
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        echo=False,
        pool_size=COMBO_CONCURRENCY + 2,
        max_overflow=5,
    )
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    all_results = load_checkpoint(fresh)
    done_keys = {
        _combo_key(
            r["country_id"], r["travel_purpose"], r["visa_type"], r["age_band"]
        )
        for r in all_results
    }
    if done_keys:
        print(
            f"♻️ 체크포인트 로드: 완료된 조합 {len(done_keys)}개는 건너뜁니다. "
            "(처음부터 다시 하려면 --fresh)"
        )

    async with async_session() as db:
        countries = await get_all_countries(db)
    if not countries:
        print("❌ 대상 국가가 없습니다.")
        await engine.dispose()
        return

    print(f"📋 대상 국가: {[c['country_name'] for c in countries]}")
    combinations = list(
        itertools.product(countries, TRAVEL_PURPOSES, VISA_TYPES, AGE_BANDS)
    )
    if limit:
        combinations = combinations[:limit]
    total = len(combinations)
    print(f"📊 총 조합 수: {total}")

    # 생성 대상(별칭 아님 + 미완료) 조합만 추려서 동시 처리한다
    todo = [
        (country, purpose, visa, age)
        for country, purpose, visa, age in combinations
        if _combo_key(country["country_id"], purpose, visa, age) not in done_keys
        and resolve_canonical(purpose, visa, age) == (purpose, visa, age)
    ]
    print(
        f"🧮 생성 대상 조합: {len(todo)}개 "
        f"(동시 조합 {COMBO_CONCURRENCY}개 / LLM 동시 {GLOBAL_LLM_CONCURRENCY}개)"
    )

    save_lock = asyncio.Lock()
    combo_sem = asyncio.Semaphore(COMBO_CONCURRENCY)
    progress = {"done": 0}

    async def worker(country, purpose, visa, age):
        cid, cname = country["country_id"], country["country_name"]
        async with combo_sem:
            try:
                result = await process_combination(
                    cid, cname, purpose, visa, age, async_session
                )
            except Exception as e:
                # 조합 하나가 실패해도 배치는 계속 — 미저장 조합은 재실행 시 자동 재시도
                print(f"  ❌ [{cname}|{purpose}|{visa}|{age}] 처리 실패 (재실행 시 재시도): {e}")
                return
        if result is None:
            return
        async with save_lock:
            all_results.append(result)
            save_checkpoint(all_results)
            progress["done"] += 1
            print(
                f"  💾 [{progress['done']}/{len(todo)}] "
                f"{cname} | {purpose} | {visa} | {age} → 카드 {len(result['risk_list'])}개"
            )

    if todo:
        await asyncio.gather(*[worker(*c) for c in todo])

    # ── 별칭 조합 채우기: 원본 조합의 카드를 그대로 복사 ──
    results_by_key = {
        _combo_key(
            r["country_id"], r["travel_purpose"], r["visa_type"], r["age_band"]
        ): r
        for r in all_results
    }
    alias_added = 0
    for country, purpose, visa, age in combinations:
        cid = country["country_id"]
        if _combo_key(cid, purpose, visa, age) in results_by_key:
            continue
        canon = resolve_canonical(purpose, visa, age)
        if canon == (purpose, visa, age):
            continue  # 별칭이 아닌데 결과가 없는 조합 = 생성 실패 (재실행 시 재시도)
        source = results_by_key.get(_combo_key(cid, *canon))
        if source is None:
            print(f"  ⚠️ 별칭 원본 미생성: {cid}|{purpose}|{visa}|{age} ← {canon}")
            continue
        alias_entry = {
            "country_id": cid,
            "travel_purpose": purpose,
            "visa_type": visa,
            "age_band": age,
            "overall_risk_level": source["overall_risk_level"],
            "risk_list": copy.deepcopy(source["risk_list"]),
            "_meta": {
                "generated_at": datetime.now().isoformat(),
                "alias_of": {
                    "travel_purpose": canon[0],
                    "visa_type": canon[1],
                    "age_band": canon[2],
                },
            },
        }
        all_results.append(alias_entry)
        results_by_key[_combo_key(cid, purpose, visa, age)] = alias_entry
        alias_added += 1
    if alias_added:
        save_checkpoint(all_results)
        print(f"\n🔗 별칭 조합 {alias_added}개 복사 완료")

    total_cards = sum(len(r["risk_list"]) for r in all_results)
    total_filtered = sum(
        r.get("_meta", {}).get("filtered_count", 0) for r in all_results
    )
    print(f"\n💾 저장 완료: {OUTPUT_FILE}")
    print(f"📊 총 {len(all_results)}개 조합, {total_cards}개 카드")
    print(f"   제거된 카드: {total_filtered}개")
    await engine.dispose()
    print("🎉 배치 생성 완료!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="리스크 카드 배치 생성")
    p.add_argument("--limit", type=int, default=None, help="테스트용 최대 조합 수")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="기존 결과 파일(체크포인트)을 무시하고 처음부터 생성",
    )
    args = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(limit=args.limit, fresh=args.fresh))
