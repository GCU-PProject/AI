# src/scripts/insert_dummy.py

import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import vertexai
from dotenv import load_dotenv

from src.core.database import Base
from src.core.models import TestLaw, Country

load_dotenv()
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# 초기 세팅할 국가 데이터 (더미 데이터 JSON의 country_id와 매칭되어야 함)
# 1: 한국, 2: 영국, 3: 싱가포르
INITIAL_COUNTRIES = [
    {"country_id": 1, "country_code": "KR", "country_name": "대한민국"},
    {"country_id": 2, "country_code": "GB", "country_name": "영국"},
    {"country_id": 3, "country_code": "SG", "country_name": "싱가포르"},
]


async def insert_data():
    print("🔄 GCP Vertex AI 연결 중...")
    try:
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION")
        )
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        print("✅ Vertex AI 연결 성공")
    except Exception as e:
        print(f"❌ Vertex AI 연결 실패: {e}")
        return

    engine = create_async_engine(DATABASE_URL, echo=False)

    # 1. DB 초기화 (기존 테이블 삭제 및 재생성)
    print("🔄 DB 테이블 초기화 중 (기존 데이터 삭제)...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # 의존성 때문에 test_laws 먼저 삭제하고 test_countries 삭제해야 함 (Drop)
        # 하지만 drop_all은 의존성을 알아서 처리해줍니다.
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        print("✅ DB 테이블(test_countries, test_laws) 재생성 완료")

    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # 2. JSON 파일 읽기
    json_path = "src/data/dummy_laws.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dummy_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {json_path} 파일을 찾을 수 없습니다.")
        return

    async with async_session() as session:
        # 3. [신규] 국가 데이터 먼저 삽입
        print("🚀 국가 데이터(test_countries) 주입 중...")
        for c in INITIAL_COUNTRIES:
            new_country = Country(
                country_id=c["country_id"],  # ID 강제 지정 (JSON과 매칭 위해)
                country_code=c["country_code"],
                country_name=c["country_name"],
            )
            session.add(new_country)
        await session.commit()
        print("✅ 국가 데이터 주입 완료")

        # 4. 법률 데이터 주입
        print(f"🚀 법률 데이터 주입 시작 (총 {len(dummy_data)}개)...")
        for data in dummy_data:
            # 임베딩 생성
            text_input = TextEmbeddingInput(
                text=data["content"], task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = embedding_model.get_embeddings([text_input])
            vector = embeddings[0].values

            e_date = (
                datetime.strptime(data["enactment_date"], "%Y-%m-%d")
                if data.get("enactment_date")
                else None
            )
            a_date = (
                datetime.strptime(data["amendment_date"], "%Y-%m-%d")
                if data.get("amendment_date")
                else None
            )

            new_law = TestLaw(
                country_id=data[
                    "country_id"
                ],  # 이제 이 ID는 countries 테이블에 반드시 존재해야 함
                law_title=data["law_title"],
                category=data.get("category"),
                article_no=data["article_no"],
                content=data["content"],
                enactment_date=e_date,
                amendment_date=a_date,
                embedding=vector,
            )
            session.add(new_law)
            print(
                f"➕ 추가됨: [{data['country_id']}] {data['law_title']} - {data['article_no']}"
            )

        await session.commit()
        print("\n🎉 모든 데이터 주입이 완료되었습니다!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(insert_data())
