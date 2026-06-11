# src/scripts/data_check_distance.py

import asyncio
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

# 1. 프로젝트 루트 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.config import settings
from src.models import Law

# chat_service에서 임베딩 모델과 임계값 설정을 가져옵니다. (로직 일치 보장)
from src.services.chat_service import MAX_DISTANCE_THRESHOLD, embeddings

load_dotenv()

# ==========================================
# 🧪 테스트 설정 (여기를 바꿔가며 실험하세요)
# ==========================================
TEST_QUERY = "Is it okay to drink alcohol in California?"
TEST_COUNTRY_ID = 2  # 1=미국연방, 2=캘리포니아, 3=뉴욕, 4=캐나다연방, 5=온타리오, 6=BC
TEST_LIMIT = 100  # 상위 몇 개까지 볼 것인지 (Top-K보다 넉넉하게 설정)
# ==========================================


async def check_distance():
    print(
        f"🔄 분석 시작... 질문: '{TEST_QUERY}' (Target Country ID: {TEST_COUNTRY_ID})"
    )

    # 1. DB 엔진 생성
    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=False)

    # 2. 질문 임베딩 (chat_service와 동일한 방식)
    try:
        query_vector = embeddings.embed_query(TEST_QUERY)
    except (ValueError, RuntimeError) as e:
        print(f"❌ 임베딩 실패: {e}")
        return

    async with engine.connect() as conn:
        # 4. rag_service와 동일한 검색 쿼리 + 동일한 필터링
        stmt = (
            select(
                Law.law_id,
                Law.law_type,
                Law.section_title,
                Law.article_no,
                Law.content,
                Law.embedding.l2_distance(query_vector).label("distance"),
            )
            .where(Law.country_id == TEST_COUNTRY_ID)  # ✅ 국가 필터링 적용
            .order_by(Law.embedding.l2_distance(query_vector))
            .limit(TEST_LIMIT)  # 테스트를 위해 넉넉하게 조회
        )

        result = await conn.execute(stmt)
        rows = result.all()

        # 5. 결과 시각화 출력
        print("\n" + "=" * 100)
        print(f"🔎 질문: {TEST_QUERY}")
        print(f"🎯 현재 설정된 임계값(Threshold): {MAX_DISTANCE_THRESHOLD}")
        print(f"🌍 필터링 국가 ID: {TEST_COUNTRY_ID}")
        print("=" * 100)
        print(
            f"{'Rank':<5} | {'ID':<6} | {'Distance':<10} | {'Status':<10} | {'Law Info':<25} | {'Content Preview'}"
        )
        print("-" * 100)

        for i, row in enumerate(rows):
            law_id = row[0]
            law_type = row[1]
            article_no = row[3]
            # 보기 좋게 줄바꿈 제거 및 길이 제한
            raw_content = row[4] or ""
            content = raw_content[:40].replace("\n", " ") + "..."
            distance = row[5]

            # 시각적 표시 (PASS / FAIL)
            if distance <= MAX_DISTANCE_THRESHOLD:
                status = "✅ PASS"  # RAG에 사용될 문서
                color_start = "\033[92m"  # 초록색 (터미널 지원 시)
            else:
                status = "❌ FAIL"  # 버려질 문서
                color_start = "\033[91m"  # 빨간색

            color_end = "\033[0m"

            print(
                f"{color_start}{i + 1:<5} | {law_id:<6} | {distance:.5f}    | {status:<10} | [{law_type}] {article_no:<10} | {content}{color_end}"
            )

        print("=" * 100)

        if not rows:
            print(
                f"⚠️ 검색 결과가 없습니다. (ID {TEST_COUNTRY_ID}에 해당하는 데이터가 없거나 DB 연결 문제)"
            )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(check_distance())
