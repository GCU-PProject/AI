"""
임베딩 서버 호환성 검증 스크립트
───────────────────────────────────────────────────────────────────
[목적]
  새로 띄운 임베딩 서버가 DB에 저장된 벡터와 "같은 방식"으로 임베딩하는지 검증한다.
  DB의 법률 한 건을 골라, 그 본문을 임베딩 서버에 보내 나온 벡터와
  DB에 저장된 벡터의 코사인 유사도를 계산한다.

  → 유사도가 0.999 이상이면 완벽히 일치 (검색 정상 작동 보장)
  → 0.9 미만이면 임베딩 방식 불일치 (접두사/pooling/모델 확인 필요)

[실행] (FastAPI VM 또는 DB에 접근 가능한 환경에서)
  EMBEDDING_URL=http://<임베딩_VM_내부IP>:8081 uv run python -m src.scripts.embed_verify_server
"""

import asyncio
import os

import httpx
import numpy as np
from sqlalchemy import select

from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.models import Law

# 기본값은 settings.EMBEDDING_URL(=.env 또는 config), 필요 시 환경변수로 덮어쓰기 가능
EMBEDDING_URL = os.getenv("EMBEDDING_URL", settings.EMBEDDING_URL)


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def main():
    print(f"🔗 임베딩 서버: {EMBEDDING_URL}")

    # 1) DB에서 임베딩이 있는 법률 한 건 조회
    async with AsyncSessionLocal() as db:
        row = (
            await db.execute(select(Law).where(Law.embedding.isnot(None)).limit(1))
        ).scalar_one_or_none()

    if row is None:
        print("🚨 임베딩이 저장된 법률이 없습니다.")
        return

    stored_vec = list(row.embedding)
    print(f"📄 검증 대상 law_id={row.law_id}, 저장된 벡터 차원={len(stored_vec)}")

    # 2) 데이터 임베딩 때와 동일한 텍스트 구성 (data_embed_to_file_qwen.py와 일치)
    text = f"{row.section_title or ''} {row.article_no or ''} {row.content or ''}"

    # 3) 임베딩 서버 호출
    async with httpx.AsyncClient(timeout=60) as client:
        health = await client.get(f"{EMBEDDING_URL}/health")
        print(f"🩺 health: {health.json()}")

        resp = await client.post(f"{EMBEDDING_URL}/embed", json={"inputs": text})
        resp.raise_for_status()
        server_vec = resp.json()[0]

    print(f"🧮 서버가 반환한 벡터 차원={len(server_vec)}")

    # 4) 차원 일치 확인
    if len(server_vec) != len(stored_vec):
        print(f"❌ 차원 불일치! 서버={len(server_vec)} vs DB={len(stored_vec)}")
        return

    # 5) 코사인 유사도 비교
    sim = cosine(server_vec, stored_vec)
    print(f"\n📊 코사인 유사도: {sim:.6f}")
    if sim >= 0.999:
        print("✅ 완벽 일치 — 임베딩 서버가 DB 벡터와 동일하게 동작합니다.")
    elif sim >= 0.9:
        print("⚠️  거의 일치하나 미세 차이 존재 (모델 버전/dtype 차이 가능). 검색엔 대체로 OK.")
    else:
        print("❌ 불일치! 접두사/pooling/모델 설정이 데이터 임베딩 때와 다릅니다.")


if __name__ == "__main__":
    asyncio.run(main())
