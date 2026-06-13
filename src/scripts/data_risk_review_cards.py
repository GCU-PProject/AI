"""
data_risk_review_cards.py

리스크 카드 자동 검수(팩트체크) 스크립트

data_risk_generate_cards.py가 만든 risk_cards.json의 각 카드를
'인용된 법령 원문'과 대조해 아래 오류를 판정하고, 있으면 자동 교정한다.
  1. 처벌 주체 오인 (고용주↔본인 등)
  2. 근거 없는 수치 (법령에 없는 벌금액·징역 기간)
  3. 확대 해석
  4. 속인주의 오적용 (마약·도박·성범죄·총기·세관 외 분야에 한국법 언급)

[특징]
- 판정 LLM이 검사 + 교정을 한 번에 수행한다.
- 동일 내용 카드(별칭 복사본 등)는 캐시로 재사용 → 중복 LLM 호출 없음.
- 카드 단위 검수 표시(_review) → 중단 후 재실행 시 이어서 진행.
- 교정된 카드 목록을 마지막에 출력 → 사람이 샘플만 확인하면 된다.

실행:
python -m src.scripts.data_risk_review_cards
python -m src.scripts.data_risk_review_cards --fresh   # 검수 기록 무시하고 전체 재검수
python -m src.scripts.data_risk_review_cards --limit 5 # 앞 N개 조합만 (테스트)
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.models import Law


# =========================================================
# 1. 설정
# =========================================================
REVIEW_MODEL = "gemini-3.5-flash"
VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}
LEVEL_PRIORITY = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# 전체 동시 판정 LLM 호출 상한 (429 rate limit 발생 시 낮추세요)
GLOBAL_LLM_CONCURRENCY = 12

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARDS_FILE = os.path.join(BASE_DIR, "data", "risk_cards.json")


# =========================================================
# 2. LLM + 프롬프트
# =========================================================
llm = ChatGoogleGenerativeAI(
    model=REVIEW_MODEL,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
    temperature=0,
    max_tokens=8192,
)

parser = JsonOutputParser()

review_yaml = load_prompt("src/prompts/risk_card_review.yaml", encoding="utf-8")
REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", review_yaml.template),
        ("human", "위 카드를 검증하고 결과를 JSON으로 반환하세요."),
    ]
)


# 전역 동시 LLM 호출 제한 — 병렬 검수 시 rate limit(429)을 방어한다.
_llm_semaphore = asyncio.Semaphore(GLOBAL_LLM_CONCURRENCY)


async def _ainvoke_with_retry(chain, inputs: dict, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            async with _llm_semaphore:
                return await chain.ainvoke(inputs)
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(2)


# =========================================================
# 3. 법령 원문 조회 / 캐시 키
# =========================================================


async def fetch_law_texts(law_ids: list[int], db: AsyncSession) -> dict[int, Law]:
    ids = [i for i in law_ids if isinstance(i, int)]
    if not ids:
        return {}
    result = await db.execute(select(Law).where(Law.law_id.in_(ids)))
    return {law.law_id: law for law in result.scalars().all()}


def format_law_context(law_refs: list[dict], law_map: dict[int, Law]) -> str:
    parts = []
    for ref in law_refs:
        law = law_map.get(ref.get("law_id"))
        if law is None:
            continue
        parts.append(
            f"[{law.law_type} {law.article_no}] {law.section_title or ''}\n{law.content}"
        )
    return "\n---\n".join(parts)


def card_cache_key(card: dict) -> str:
    """동일 내용 카드(별칭 복사본 등)를 식별하는 키. 교정 적용 전 원본 기준으로 계산해야 한다."""
    law_ids = ",".join(
        str(r.get("law_id")) for r in sorted(
            card.get("law_refs", []), key=lambda r: str(r.get("law_id"))
        )
    )
    raw = f"{card.get('risk_title', '')}\n{card.get('risk_content', '')}\n{law_ids}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# =========================================================
# 4. 카드 1장 검수
# =========================================================


async def run_judge(card: dict, law_context: str) -> dict:
    """판정 LLM 호출 (DB 미사용 — 병렬 호출 가능). 결과 dict를 반환한다."""
    chain = REVIEW_PROMPT | llm | parser
    result = await _ainvoke_with_retry(
        chain,
        {
            "risk_title": card.get("risk_title", ""),
            "risk_level": card.get("risk_level", ""),
            "risk_content": card.get("risk_content", ""),
            "risk_actions": json.dumps(
                card.get("risk_actions", []), ensure_ascii=False
            ),
            "law_context": law_context,
        },
    )
    if not isinstance(result, dict) or result.get("verdict") not in {"ok", "fix"}:
        return {"verdict": "unverifiable"}
    return result


def apply_result(card: dict, result: dict) -> str:
    """판정 결과를 카드에 반영하고 verdict를 반환한다."""
    verdict = result.get("verdict", "unverifiable")
    if verdict == "fix":
        new_title = result.get("risk_title", "").strip()
        if new_title:
            card["risk_title"] = new_title
        new_content = result.get("risk_content", "").strip()
        if new_content:
            card["risk_content"] = new_content
        new_actions = result.get("risk_actions")
        if isinstance(new_actions, list) and new_actions:
            card["risk_actions"] = new_actions
        new_level = result.get("risk_level", "")
        if new_level in VALID_LEVELS:
            card["risk_level"] = new_level
    card["_review"] = {"verdict": verdict, "issues": result.get("issues", [])}
    return verdict


def recalc_overall(entry: dict) -> None:
    cards = entry.get("risk_list", [])
    if not cards:
        entry["overall_risk_level"] = "LOW"
        return
    max_p = max(LEVEL_PRIORITY.get(c.get("risk_level"), 1) for c in cards)
    entry["overall_risk_level"] = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}[max_p]


# =========================================================
# 5. 체크포인트
# =========================================================


def save_cards(data: list[dict]) -> None:
    tmp = CARDS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CARDS_FILE)


# =========================================================
# 6. 메인
# =========================================================


async def main(limit: int | None = None, fresh: bool = False, chunk: int = 50):
    print("🔎 리스크 카드 자동 검수 시작")
    print(f"🤖 판정 모델: {REVIEW_MODEL} (동시 {GLOBAL_LLM_CONCURRENCY}콜)")

    if not os.path.exists(CARDS_FILE):
        print(f"❌ 파일이 없습니다: {CARDS_FILE}")
        return
    with open(CARDS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    # 검수는 앞 N개 조합만 (테스트). 나머지 조합은 그대로 보존.
    target_entries = data[:limit] if limit else data

    # 1) 검수 대상 카드를 고유 키로 묶는다 (별칭 복사본은 한 번만 판정 → 결과 공유)
    key_to_cards: dict[str, list[dict]] = {}
    key_to_rep: dict[str, dict] = {}
    for entry in target_entries:
        for card in entry.get("risk_list", []):
            if card.get("_review") and not fresh:
                continue  # 이미 검수됨 (이어하기)
            key = card_cache_key(card)
            key_to_cards.setdefault(key, []).append(card)
            key_to_rep.setdefault(key, card)

    unique_keys = list(key_to_rep)
    total_cards = sum(len(v) for v in key_to_cards.values())
    print(
        f"🧮 검수 대상: 고유 카드 {len(unique_keys)}개 "
        f"(전체 {total_cards}장, 별칭 중복 {total_cards - len(unique_keys)}장은 결과 공유)"
    )
    if not unique_keys:
        print("✅ 검수할 카드가 없습니다. (모두 검수 완료 상태)")
        return

    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # 2) 필요한 법령 원문을 일괄 조회 (DB 1회, 이후 판정은 DB 미사용)
    all_law_ids = {
        r.get("law_id")
        for key in unique_keys
        for r in key_to_rep[key].get("law_refs", [])
        if isinstance(r.get("law_id"), int)
    }
    async with async_session() as db:
        law_map = await fetch_law_texts(list(all_law_ids), db)
    await engine.dispose()
    print(f"📚 법령 원문 {len(law_map)}건 로드 완료")

    counts = {"ok": 0, "fix": 0, "unverifiable": 0}
    fixed_log: list[str] = []

    async def judge_key(key: str) -> tuple[str, dict]:
        rep = key_to_rep[key]
        law_context = format_law_context(rep.get("law_refs", []), law_map)
        if not law_context.strip():
            return key, {"verdict": "unverifiable"}
        try:
            return key, await run_judge(rep, law_context)
        except Exception as ex:
            print(f"  ⚠️ 판정 실패(unverifiable 처리): {ex}")
            return key, {"verdict": "unverifiable"}

    # 3) 청크 단위 병렬 판정 → 청크마다 결과 적용 + 저장 (이어하기 보장)
    for i in range(0, len(unique_keys), chunk):
        batch = unique_keys[i : i + chunk]
        results = await asyncio.gather(*[judge_key(k) for k in batch])
        for key, result in results:
            verdict = result.get("verdict", "unverifiable")
            for card in key_to_cards[key]:  # 별칭 복사본까지 동일 결과 적용
                apply_result(card, result)
            n = len(key_to_cards[key])
            counts[verdict] = counts.get(verdict, 0) + n
            if verdict == "fix":
                rep = key_to_rep[key]
                fixed_log.append(rep.get("risk_title", "")[:50])
        # 영향받은 조합의 overall 재계산 후 저장
        for entry in target_entries:
            recalc_overall(entry)
        save_cards(data)
        done = min(i + chunk, len(unique_keys))
        print(
            f"  [{done}/{len(unique_keys)}] 판정 진행 — "
            f"ok {counts['ok']} / fix {counts['fix']} / unverifiable {counts['unverifiable']}"
        )

    print("\n" + "=" * 55)
    print("📊 검수 결과 (카드 장 수 기준)")
    print(f"   정상(ok):    {counts['ok']}")
    print(f"   교정(fix):   {counts['fix']}")
    print(f"   검증불가:    {counts['unverifiable']} (인용 법령 원문 없음)")
    print("=" * 55)
    if fixed_log:
        print(f"\n✏️ 교정된 카드 {len(fixed_log)}개 (샘플 확인용):")
        for line in fixed_log[:40]:
            print(f"   - {line}")
        if len(fixed_log) > 40:
            print(f"   ... 외 {len(fixed_log) - 40}개")
    print("\n🎉 검수 완료! (결과는 risk_cards.json에 반영됨)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="리스크 카드 자동 검수")
    p.add_argument("--limit", type=int, default=None, help="앞 N개 조합만 검수 (테스트)")
    p.add_argument(
        "--fresh", action="store_true", help="기존 검수 기록을 무시하고 전체 재검수"
    )
    args = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(limit=args.limit, fresh=args.fresh))
