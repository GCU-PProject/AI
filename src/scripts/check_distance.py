# src/scripts/check_distance.py

import asyncio
import os
import sys
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select
from vertexai.language_models import TextEmbeddingInput

# 1. 프로젝트 루트 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.models import Law
from src.core.config import settings

# ✅ rag_service에서 모델 로드 함수와 설정을 그대로 가져옵니다. (로직 일치 보장)
from services.chat_service import get_models, MAX_DISTANCE_THRESHOLD

load_dotenv()

# ==========================================
# 🧪 테스트 설정 (여기를 바꿔가며 실험하세요)
# ==========================================
TEST_QUERY = "음주운전 처벌 기준이 뭐야?"
TEST_COUNTRY_ID = 1  # 1: 한국, 2: 영국, 3: 싱가포르
TEST_LIMIT = 10  # 상위 몇 개까지 볼 것인지 (Top-K보다 넉넉하게 설정)
# ==========================================


async def check_distance():
    print(
        f"🔄 분석 시작... 질문: '{TEST_QUERY}' (Target Country ID: {TEST_COUNTRY_ID})"
    )

    # 1. rag_service와 동일한 모델 로드 함수 사용
    embedding_model, _ = get_models()

    # 2. DB 엔진 생성
    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    # 3. 질문 임베딩 (rag_service와 동일한 방식)
    try:
        text_input = TextEmbeddingInput(text=TEST_QUERY, task_type="RETRIEVAL_QUERY")
        embeddings = embedding_model.get_embeddings([text_input])
        query_vector = embeddings[0].values
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return

    async with engine.connect() as conn:
        # 4. rag_service와 동일한 검색 쿼리 + 동일한 필터링
        stmt = (
            select(
                Law.law_title,
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
            f"{'Rank':<5} | {'Distance':<10} | {'Status':<10} | {'Law Info':<25} | {'Content Preview'}"
        )
        print("-" * 100)

        for i, row in enumerate(rows):
            law_title = row[0]
            article_no = row[1]
            # 보기 좋게 줄바꿈 제거 및 길이 제한
            content = row[2][:40].replace("\n", " ") + "..."
            distance = row[3]

            # 시각적 표시 (PASS / FAIL)
            if distance <= MAX_DISTANCE_THRESHOLD:
                status = "✅ PASS"  # RAG에 사용될 문서
                color_start = "\033[92m"  # 초록색 (터미널 지원 시)
            else:
                status = "❌ FAIL"  # 버려질 문서
                color_start = "\033[91m"  # 빨간색

            color_end = "\033[0m"

            print(
                f"{color_start}{i+1:<5} | {distance:.5f}    | {status:<10} | {law_title} {article_no:<10} | {content}{color_end}"
            )

        print("=" * 100)

        if not rows:
            print(
                f"⚠️ 검색 결과가 없습니다. (ID {TEST_COUNTRY_ID}에 해당하는 데이터가 없거나 DB 연결 문제)"
            )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(check_distance())
