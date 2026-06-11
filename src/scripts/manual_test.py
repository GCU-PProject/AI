"""
수동 품질 테스트 스크립트 — 막연한 질문 / 엣지케이스 / 정상케이스

[실행]
  uv run python -m src.scripts.manual_test

[설정]
  BASE_URL: FastAPI 서버 주소 (기본 localhost:8000)
"""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"

# country_id: 1=미국연방, 2=캘리포니아, 3=뉴욕, 4=캐나다연방, 5=온타리오, 6=BC
TESTS = [
    # ── 캘리포니아 ───────────────────────────────────────────
    {"country": "캘리포니아", "country_id": 2, "type": "막연",   "query": "대마초 하면 어떻게 돼?"},
    {"country": "캘리포니아", "country_id": 2, "type": "막연",   "query": "술 마시고 운전하면 어떻게 되는 거야?"},
    {"country": "캘리포니아", "country_id": 2, "type": "막연",   "query": "세입자 권리가 뭐야?"},
    {"country": "캘리포니아", "country_id": 2, "type": "정상",   "query": "음주운전 벌금이 얼마야?"},
    {"country": "캘리포니아", "country_id": 2, "type": "정상",   "query": "집주인이 갑자기 집에서 나가라고 하면 어떻게 해야 해?"},
    {"country": "캘리포니아", "country_id": 2, "type": "엣지",   "query": "한국에서 운전면허를 땄는데 캘리포니아에서 운전해도 돼?"},
    {"country": "캘리포니아", "country_id": 2, "type": "엣지",   "query": "오늘 날씨 어때?"},

    # ── 뉴욕 ─────────────────────────────────────────────────
    {"country": "뉴욕",       "country_id": 3, "type": "막연",   "query": "총기 소지하면 어떻게 돼?"},
    {"country": "뉴욕",       "country_id": 3, "type": "막연",   "query": "직장에서 해고당하면 어떻게 해?"},
    {"country": "뉴욕",       "country_id": 3, "type": "정상",   "query": "최저임금이 얼마야?"},
    {"country": "뉴욕",       "country_id": 3, "type": "엣지",   "query": "뉴욕에서 대마초 피워도 돼?"},

    # ── 캐나다 연방 ───────────────────────────────────────────
    {"country": "캐나다연방",  "country_id": 4, "type": "막연",   "query": "이민 오면 어떻게 해야 해?"},
    {"country": "캐나다연방",  "country_id": 4, "type": "정상",   "query": "캐나다에서 대마초 합법이야?"},
    {"country": "캐나다연방",  "country_id": 4, "type": "엣지",   "query": "캘리포니아 법이랑 캐나다 법 중 뭐가 더 엄격해?"},

    # ── 온타리오 ─────────────────────────────────────────────
    {"country": "온타리오",   "country_id": 5, "type": "막연",   "query": "집 사면 어떤 세금 내야 해?"},
    {"country": "온타리오",   "country_id": 5, "type": "정상",   "query": "집주인이 보증금을 안 돌려주면 어떻게 해?"},

    # ── BC ───────────────────────────────────────────────────
    {"country": "BC",         "country_id": 6, "type": "막연",   "query": "사고 나면 어떻게 해야 해?"},
    {"country": "BC",         "country_id": 6, "type": "정상",   "query": "BC에서 음주운전 걸리면 면허 정지돼?"},
]


async def run_test(client: httpx.AsyncClient, test: dict) -> dict:
    try:
        resp = await client.post(
            f"{BASE_URL}/api/qna",
            json={"query": test["query"], "country_id": test["country_id"]},
            timeout=30,
        )
        print(f"  [HTTP {resp.status_code}] {resp.text[:200]}")
        data = resp.json()
        result = data.get("result") or {}
        return {
            **test,
            "success": result.get("search_success", False),
            "law_ids": result.get("related_law_id_list", []),
            "answer": result.get("answer") or data.get("message", "오류"),
        }
    except Exception as e:
        return {**test, "success": False, "law_ids": [], "answer": f"[요청 실패] {e}"}


def print_result(r: dict):
    tag = {"막연": "🌫 막연", "정상": "✅ 정상", "엣지": "⚠️  엣지"}[r["type"]]
    search = "🔍 검색성공" if r["success"] else "❌ 검색실패"
    print(f"\n{'─'*70}")
    print(f"[{r['country']}] {tag} | {search}")
    print(f"Q: {r['query']}")
    print(f"참조법령: {r['law_ids']}")
    print(f"A: {r['answer'][:300]}{'...' if len(r['answer']) > 300 else ''}")


async def main():
    print(f"🚀 테스트 시작 — {len(TESTS)}개 질문 / 서버: {BASE_URL}")
    async with httpx.AsyncClient() as client:
        results = []
        for test in TESTS:
            r = await run_test(client, test)
            print_result(r)
            results.append(r)

    # 요약
    total = len(results)
    search_ok = sum(1 for r in results if r["success"])
    print(f"\n{'='*70}")
    print(f"📊 검색성공: {search_ok}/{total}")
    for t in ["막연", "정상", "엣지"]:
        group = [r for r in results if r["type"] == t]
        ok = sum(1 for r in group if r["success"])
        print(f"  {t}: {ok}/{len(group)} 검색성공")


if __name__ == "__main__":
    asyncio.run(main())
