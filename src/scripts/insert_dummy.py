# src/scripts/insert_dummy.py

import asyncio
import json
import os
import sys
from datetime import datetime

# 1. 현재 파일의 위치를 기준으로 상위 폴더(src)를 모듈 경로에 추가
# (이게 없으면 src 폴더 안의 다른 파일을 못 불러옵니다)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    BigInteger,
    text,
    select,
    func,
)
from pgvector.sqlalchemy import Vector
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import vertexai
from dotenv import load_dotenv

from src.core.database import Base
from src.core.models import Law

# .env 파일 로드 (DB 정보, GCP 정보 가져오기)
load_dotenv()

# DB 접속 URL 생성
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


async def insert_data():
    print("🔄 GCP Vertex AI 연결 중...")
    # 3. GCP Vertex AI 연결
    try:
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION")
        )
        # 구글의 최신 한국어/영어 지원 임베딩 모델
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        print("✅ Vertex AI 연결 성공")
    except Exception as e:
        print(f"❌ Vertex AI 연결 실패: {e}")
        print("💡 힌트: gcp-key.json 파일이 있고, 환경변수 설정이 되었는지 확인하세요.")
        return

    # 4. DB 연결 엔진 시작
    engine = create_async_engine(DATABASE_URL, echo=False)

    # 테이블 생성 (없을 경우에만)
    async with engine.begin() as conn:
        # pgvector 확장 기능 켜기
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # 테이블 만들기
        await conn.run_sync(Base.metadata.create_all)
        print("✅ DB 테이블 준비 완료 (laws 테이블)")

    # 세션 생성
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # 5. JSON 파일 읽기
    json_path = "src/data/dummy_laws.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dummy_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {json_path} 파일을 찾을 수 없습니다.")
        return

    # 6. 데이터 주입 루프
    async with async_session() as session:
        print(f"🚀 데이터 주입 시작 (총 {len(dummy_data)}개)...")

        for data in dummy_data:
            # 중복 방지: 이미 같은 조항(article_no)이 있는지 확인
            exists = await session.execute(
                select(Law).where(Law.article_no == data["article_no"])
            )
            if exists.scalar():
                print(f"⚠️ 스킵: {data['article_no']} (이미 DB에 있음)")
                continue

            # (A) 임베딩 생성: 텍스트 -> 벡터 변환
            # task_type="RETRIEVAL_DOCUMENT"는 "이건 검색될 문서야"라고 모델에 알려주는 것
            text_input = TextEmbeddingInput(
                text=data["content"], task_type="RETRIEVAL_DOCUMENT"
            )

            embeddings = embedding_model.get_embeddings([text_input])
            vector = embeddings[0].values

            # (B) 날짜 문자열 -> 날짜 객체 변환
            # JSON에 값이 없으면 None 처리
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

            # (C) DB 객체 만들기
            new_law = Law(
                country_id=data["country_id"],
                law_title=data["law_title"],
                category=data["category"],
                article_no=data["article_no"],
                content=data["content"],
                enactment_date=e_date,
                amendment_date=a_date,
                embedding=vector,
            )

            # 세션에 추가
            session.add(new_law)
            print(f"➕ 추가됨: {data['law_title']} - {data['article_no']}")

        # 최종 저장 (Commit)
        await session.commit()
        print("\n🎉 모든 데이터 주입이 완료되었습니다!")


if __name__ == "__main__":
    asyncio.run(insert_data())
