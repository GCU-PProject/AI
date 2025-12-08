# src/scripts/analyze_threshold.py

import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import vertexai

# 1. 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.models import Law
from src.core.config import settings

# 테스트할 질문
TEST_QUERY = (
    "What are the criteria for driver's license revocation due to drunk driving"
)


async def analyze_threshold():
    print(f"🔄 분석 시작... 질문: '{TEST_QUERY}'")

    # 2. Config의 설정값 사용
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    # 3. DB 연결도 settings의 URL 사용
    # (echo=False로 설정하여 SQL 로그가 너무 많이 나오는 것 방지)
    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    # 4. 질문 임베딩
    text_input = TextEmbeddingInput(text=TEST_QUERY, task_type="RETRIEVAL_QUERY")
    embeddings = embedding_model.get_embeddings([text_input])
    query_vector = embeddings[0].values

    async with engine.connect() as conn:
        # DB 거리 계산 쿼리 (이전과 동일)
        stmt = (
            select(
                Law.law_title,
                Law.article_no,
                Law.content,
                Law.embedding.l2_distance(query_vector).label("distance"),
            )
            .order_by(Law.embedding.l2_distance(query_vector))
            .limit(20)
        )

        result = await conn.execute(stmt)
        rows = result.all()

        # 결과 출력 (이전과 동일)
        print("\n" + "=" * 80)
        print(f"🔎 질문: {TEST_QUERY}")
        print("=" * 80)
        print(f"{'Rank':<5} | {'Distance':<10} | {'Law':<15} | {'Content (Preview)'}")
        print("-" * 80)

        for i, row in enumerate(rows):
            law_title = row[0]
            article_no = row[1]
            content = row[2][:40].replace("\n", " ") + "..."
            distance = row[3]

            color = "\033[0m"
            if distance < 0.6:
                color = "\033[92m"
            elif distance < 0.75:
                color = "\033[93m"
            else:
                color = "\033[91m"

            print(
                f"{color}{i+1:<5} | {distance:.5f}    | {law_title} {article_no:<5} | {content}\033[0m"
            )

        print("=" * 80)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(analyze_threshold())
