"""
Retrieval 진단 스크립트

[목적]
검색 실패 / 엉뚱한 결과가 나오는 케이스에 대해 파이프라인 중간 결과를 출력합니다.
  - 번역 결과
  - rewrite 결과 (실제 벡터 DB에 들어가는 쿼리)
  - 임계값 필터 전 상위 N개 결과 (distance + section_title)

[실행]
  uv run python -m src.scripts.debug_retrieval
"""

import asyncio

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.llm import embeddings
from src.models import Law
from src.services.chat_service import translate_query, rewrite_for_search

# 진단 대상: 검색 실패 또는 엉뚱한 결과가 나온 케이스
CASES = [
    {"label": "[NY] 직장 해고",          "query": "직장에서 해고당하면 어떻게 해?",        "country_id": 3},
    {"label": "[CA] 대마초",             "query": "대마초 하면 어떻게 돼?",               "country_id": 2},
    {"label": "[NY] 총기 소지",          "query": "총기 소지하면 어떻게 돼?",             "country_id": 3},
    {"label": "[NY] 대마초",             "query": "뉴욕에서 대마초 피워도 돼?",           "country_id": 3},
    {"label": "[ON] 보증금",             "query": "집주인이 보증금을 안 돌려주면 어떻게 해?", "country_id": 5},
    {"label": "[CA] 음주운전 벌금",       "query": "음주운전 벌금이 얼마야?",              "country_id": 2},
    {"label": "[CA] 세입자 권리",         "query": "세입자 권리가 뭐야?",                  "country_id": 2},
]

# 임계값 필터 없이 가져올 후보 수
RAW_TOP_K = 10
THRESHOLD = settings.RAG_MAX_DISTANCE_THRESHOLD

# country_id → 연방 포함 jurisdiction ids (chat_service.py와 동일 로직)
JURISDICTION_MAP = {
    2: [2, 1],   # 캘리포니아 + 미국 연방
    3: [3, 1],   # 뉴욕 + 미국 연방
    4: [4],      # 캐나다 연방
    5: [5, 4],   # 온타리오 + 캐나다 연방
    6: [6, 4],   # BC + 캐나다 연방
}


async def diagnose(session: AsyncSession, case: dict):
    query = case["query"]
    country_id = case["country_id"]
    jurisdiction_ids = JURISDICTION_MAP.get(country_id, [country_id])

    print(f"\n{'═' * 70}")
    print(f"  {case['label']}  |  Q: {query}")
    print(f"{'═' * 70}")

    # Step 1: 번역
    translated = await translate_query(query)
    print(f"  [번역]    {translated}")

    # Step 2: rewrite
    rewritten = await rewrite_for_search(translated)
    print(f"  [rewrite] {rewritten}")

    # Step 3: 벡터 검색 (임계값 필터 없이 RAW_TOP_K개)
    query_vector = embeddings.embed_query(rewritten)

    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id.in_(jurisdiction_ids))
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(RAW_TOP_K)
    )
    rows = (await session.execute(stmt)).all()

    print(f"\n  {'DIST':>6}  {'PASS':>4}  {'LAW_ID':>8}  SECTION_TITLE")
    print(f"  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*40}")
    for row in rows:
        law, dist = row[0], row[1]
        passed = "✅" if dist <= THRESHOLD else "❌"
        title = (law.section_title or "")[:60]
        print(f"  {dist:>6.4f}  {passed:>4}  {law.law_id:>8}  {title}")

    passing = sum(1 for r in rows if r[1] <= THRESHOLD)
    print(f"\n  → 임계값({THRESHOLD}) 통과: {passing}/{len(rows)}")


async def main():
    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        for case in CASES:
            await diagnose(session, case)

    await engine.dispose()
    print(f"\n{'═' * 70}")
    print("진단 완료")


if __name__ == "__main__":
    asyncio.run(main())
