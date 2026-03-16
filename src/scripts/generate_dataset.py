# src/scripts/generate_dataset.py
"""
[RAGAS 데이터셋 생성 스크립트]

이 스크립트는 DB(PostgreSQL)에 저장된 법률 문서(chunk)들을 불러와,
RAGAS 프레임워크를 이용해 RAG 평가용 합성 데이터셋(Synthetic Dataset)을 자동 생성합니다.

- LLM & Embeddings: Vertex AI (gemini-2.0-flash, text-embedding-005)
- 소스 데이터: `laws` 테이블의 실제 법률 조항
- 목표 데이터셋 크기: 20개 (simple 60% 등)
"""

import sys
import os
import asyncio
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import select, func
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# RAGAS 관련 임포트
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.docstore import InMemoryDocumentStore

from src.core.database import AsyncSessionLocal
from src.core.models import Law
from src.core.config import settings

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
TESTSET_SIZE = 20
DOCUMENT_LIMIT = 50  # 테스트셋 생성의 재료가 될 문서 개수 (너무 많으면 느림)
OUTPUT_CSV_PATH = "data/ragas_testset.csv"

# 질문 유형 분포 (비전문가 대상 서비스이므로 simple을 60%로 높게 설정)
DISTRIBUTIONS = {simple: 0.6, reasoning: 0.2, multi_context: 0.1, conditional: 0.1}


# =========================================================
# 2. 메인 생성 로직
# =========================================================
async def generate_dataset():
    print("🚀 [1/5] 데이터베이스에서 법률 문서를 불러옵니다...")
    docs = []

    # DB에서 문서 추출 (캘리포니아 법률 랜덤 선택)
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Law)
            .where(Law.country_id == 1)
            .order_by(func.random())
            .limit(DOCUMENT_LIMIT)
        )
        result = await session.execute(stmt)
        laws = result.scalars().all()

        for law in laws:
            # LangChain Document 형식으로 변환 (RAGAS 호환용, filename 필수)
            doc = Document(
                page_content=law.content,
                metadata={
                    "filename": f"{law.law_type}_{law.article_no}",
                    "law_id": law.law_id,
                    "law_type": law.law_type,
                    "article_no": law.article_no,
                },
            )
            docs.append(doc)

    if not docs:
        print("❌ DB에서 문서를 가져오지 못했습니다. DB 연결 또는 테이블을 확인하세요.")
        return

    print(f"✅ {len(docs)}개의 법률 문서를 성공적으로 로드했습니다.")
    print("🧠 [2/5] Vertex AI 모델 및 RAGAS 컴포넌트를 초기화합니다...")

    # 생성기 및 비평기용 Vertex AI 모델 초기화
    # 추후 비평기용 모델을 더 상위 모델로 설정 예정
    llm = ChatVertexAI(
        model_name=settings.GCP_MODEL_NAME,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        temperature=0.0,  # 법률 AI의 일관성과 정확성을 위해 0.0으로 고정
    )

    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )

    # RAGAS Wrapper로 감싸기
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Keyphrase Extractor 초기화 (문서에서 핵심어 추출)
    keyphrase_extractor = KeyphraseExtractor(llm=ragas_llm)

    # InMemoryDocumentStore 초기화
    # 원문 훼손을 막기 위해 더미 분할기를 사용(분할기 객체가 필수라서)
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    dummy_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)

    docstore = InMemoryDocumentStore(
        splitter=dummy_splitter,
        embeddings=ragas_embeddings,
        extractor=keyphrase_extractor,
    )

    print("🛠️ [3/5] Testset Generator를 설정합니다...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=ragas_embeddings,
        docstore=docstore,
    )

    print(
        f"⚙️ [4/5] RAGAS 테스트셋 생성 시작! (목표: {TESTSET_SIZE}개, 시간이 걸릴 수 있습니다)"
    )
    print(f"   - 비율: Simple 60%, Reasoning 20%, Multi 10%, Cond 10%")
    try:
        testset = generator.generate_with_langchain_docs(
            documents=docs,
            test_size=TESTSET_SIZE,
            distributions=DISTRIBUTIONS,
            with_debugging_logs=True,
            language="korean",  # 한국어로 데이터셋 생성
        )
    except Exception as e:
        print(f"\n❌ 데이터셋 생성 중 오류 발생: {e}")
        return

    print(f"💾 [5/5] 생성 완료! (결과를 CSV에 저장합니다: {OUTPUT_CSV_PATH})")
    test_df = testset.to_pandas()

    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    test_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")  # 한글 깨짐 방지

    print("\n🎉 데이터셋 생성 작업이 성공적으로 종료되었습니다.")
    print("미리보기 (상위 3개):")
    print(test_df.head(3))


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(generate_dataset())
