# src/scripts/ragas_generate_dataset.py
"""
[RAGAS 데이터셋 생성 스크립트]

이 스크립트는 DB(PostgreSQL)에 저장된 법률 문서(chunk)들을 불러와,
RAGAS 프레임워크를 이용해 RAG 평가용 합성 데이터셋(Synthetic Dataset)을 자동 생성합니다.

- LLM: Vertex AI Gemini (settings.GCP_MODEL_NAME) / Embeddings: 자체 호스팅 Qwen3-Embedding-0.6B
- 소스 데이터: `laws` 테이블의 실제 법률 조항
- 목표 데이터셋 크기: 42개 (국가당 7개 × 6개국, SingleHop 70% / MultiHopAbstract 30%)
  · 일반 사용자(유학생/여행자) 대상이므로, 조항번호를 직접 지목하는 전문가형 질문을
    만드는 MultiHopSpecific은 제외하고 SingleHop + MultiHopAbstract만 사용
  · 국가별로 따로 생성 후 country_id 태깅 → 평가 시 해당 국가 법률로만 검색 가능

[RAGAS 0.4.x API 기준]
- 구버전(0.1.x)의 evolutions(simple, reasoning, multi_context, conditional)이
  신버전(0.4.x)의 synthesizers(SingleHop, MultiHopAbstract, MultiHopSpecific)로 변경됨
- InMemoryDocumentStore → KnowledgeGraph (내부 자동 생성)
- language="korean" 파라미터 → llm_context로 한국어 생성 유도
"""

import asyncio
import os
import sys

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# LangchainLLMWrapper: LangChain LLM 객체를 RAGAS 내부 형식으로 감싸는 래퍼
# - TestsetGenerator.from_langchain()은 내부에서 자동으로 래핑해줌
# - 하지만 query_distribution을 수동으로 만들 때는 직접 래핑해서 synthesizer에 넘겨야 함
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

# ====================================================================
# RAGAS 관련 임포트 (0.4.x 신버전 API)
# ====================================================================
# TestsetGenerator: 테스트셋 생성의 메인 클래스
# - 구버전: from ragas.testset.generator import TestsetGenerator
# - 신버전: from ragas.testset import TestsetGenerator (경로가 바뀜)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
)

# Synthesizers: 질문 유형(난이도)별 생성기
# - 구버전의 evolutions(simple, reasoning, multi_context, conditional)을 대체
# - SingleHopSpecific: 하나의 chunk로 답할 수 있는 단순 질문 (≈ 구버전 simple)
# - MultiHopAbstract: 여러 chunk를 종합해 추상적 개념을 추론 (≈ 구버전 reasoning)
# - MultiHopSpecific: 여러 chunk의 구체적 정보를 연결 (조항번호를 직접 지목하는 전문가형)
#   → 일반 사용자(유학생/여행자) 타깃과 맞지 않아 본 스크립트에서는 사용하지 않음
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from sqlalchemy import func, select

from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.core.llm import embeddings  # 자체 호스팅 Qwen3 임베딩 (RemoteEmbeddings)
from src.models import Law

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
DOCS_PER_COUNTRY = 30  # 국가(country_id)별로 추출할 문서 수 (질문 생성 재료)
QUESTIONS_PER_COUNTRY = 7  # 국가별 생성 질문 수 → 6개국이면 총 42문제
OUTPUT_CSV_PATH = "data/ragas_testset.csv"

# llm_context: 생성 시 LLM에 주입되는 추가 지시 텍스트 (구버전 language="korean" 대체)
# - 0.4.x에서는 language 파라미터가 제거됨
# - 자유 텍스트 슬롯이므로 ① 서비스/사용자 시나리오 ② 한국어 출력 ③ JSON 형식을 함께 지시
# - TestsetGenerator 및 각 synthesizer에 전달하면 질문/정답 생성 방향을 유도함
# - 단, 질문의 근거는 결국 DB 법률 조항(context)이므로, 조항에 없는 내용은 생성되지 않음
LLM_CONTEXT = (
    # 1) 평가 대상 서비스 & 사용자 시나리오
    "You are generating an evaluation dataset for GLAW, a legal AI assistant "
    "that helps Korean-speaking travelers and residents abroad understand "
    "foreign laws. Its users typically ask practical, real-life questions "
    "about topics such as traffic and DUI rules, visas and immigration, "
    "penalties and fines, and everyday legal issues they may face overseas. "
    "Generate realistic questions that such users would actually ask, "
    "grounded strictly in the provided legal context. "
    # 2) 언어 규칙 (반드시 한국어 출력)
    "CRITICAL MANDATORY RULE: YOU MUST GENERATE ALL QUESTIONS AND ANSWERS EXCLUSIVELY IN THE KOREAN LANGUAGE (한국어). "
    "Even if instructed to adopt an English-speaking persona or a specific English style (e.g. 'POOR_GRAMMAR'), YOU MUST TRANSLATE your final thought into highly natural Korean before outputting. NEVER use English for the generated questions or answers. "
    # 3) 출력 형식 규칙 (JSON)
    'IMPORTANT: 1) ALWAYS output valid JSON. 2) Ensure internal quotes inside JSON strings are properly escaped (e.g. \\" instead of "). '
    "3) Do not include markdown code block backticks around your JSON payload."
)


# =========================================================
# 2. 메인 생성 로직
# =========================================================
async def _fetch_docs_by_country(cid: int) -> list[Document]:
    """특정 국가(country_id)에서 DOCS_PER_COUNTRY개의 법률 조항을 랜덤 추출한다."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Law)
            .where(Law.country_id == cid)
            .order_by(func.random())
            .limit(DOCS_PER_COUNTRY)
        )
        result = await session.execute(stmt)
        laws = result.scalars().all()

    docs = []
    for law in laws:
        docs.append(
            Document(
                page_content=law.content,
                metadata={
                    "filename": f"{law.law_type}_{law.article_no}",
                    "law_id": law.law_id,
                    "country_id": law.country_id,
                    "law_type": law.law_type,
                    "article_no": law.article_no,
                },
            )
        )
    return docs


async def generate_dataset():
    print("🚀 [1/4] 데이터가 존재하는 국가 목록을 조회합니다...")
    async with AsyncSessionLocal() as session:
        country_rows = await session.execute(
            select(Law.country_id).distinct().order_by(Law.country_id)
        )
        country_ids = [row[0] for row in country_rows]

    if not country_ids:
        print("❌ DB에서 국가 정보를 가져오지 못했습니다. DB 연결/적재 상태를 확인하세요.")
        return
    print(f"   📚 데이터가 존재하는 국가: {country_ids}")

    # ── LLM / Generator 초기화 (국가 반복과 무관하게 1회만) ──
    print("🧠 [2/4] LLM 및 RAGAS 컴포넌트를 초기화합니다...")
    llm = ChatGoogleGenerativeAI(
        model=settings.GCP_MODEL_NAME,  # 실제 서비스와 동일한 생성 모델
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        vertexai=True,
        temperature=0.0,
    )
    # 임베딩: 실제 서비스/DB와 동일한 자체 호스팅 Qwen3(RemoteEmbeddings)
    # ⚠️ 임베딩 서버(EMBEDDING_URL, 내부 IP) 접근 가능한 환경(VPC 내부/터널)에서 실행할 것.
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    ragas_llm = LangchainLLMWrapper(llm)

    # 질문 유형: 일반 사용자 타깃 → SingleHop 70% + MultiHopAbstract 30%
    # (조항번호를 지목하는 전문가형 MultiHopSpecific은 제외)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm, llm_context=LLM_CONTEXT), 0.7),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm, llm_context=LLM_CONTEXT), 0.3),
    ]

    # ── 국가별로 따로 생성 → country_id 태깅 ──
    # 각 질문이 어느 국가 법률에서 나왔는지 알 수 있어야, 평가 단계에서
    # 해당 국가(country_id)로만 검색하여 실제 서비스와 동일하게 평가할 수 있다.
    print(
        f"⚙️ [3/4] 국가별 테스트셋 생성 시작! "
        f"(국가당 {QUESTIONS_PER_COUNTRY}개 → 총 {len(country_ids) * QUESTIONS_PER_COUNTRY}개 목표)"
    )
    print("   - 비율: SingleHop 70%, MultiHopAbstract 30%")

    all_dfs = []
    for cid in country_ids:
        print(f"\n   🌍 country_id={cid} 처리 중...")
        docs = await _fetch_docs_by_country(cid)
        print(f"      - 재료 문서 {len(docs)}개 추출")
        if len(docs) < 2:
            print(f"      ⚠️ 문서가 너무 적어 건너뜁니다 (country_id={cid}).")
            continue

        try:
            testset = generator.generate_with_chunks(
                chunks=docs,
                testset_size=QUESTIONS_PER_COUNTRY,
                query_distribution=query_distribution,
                with_debugging_logs=False,
                run_config=RunConfig(timeout=120, max_retries=10, max_workers=4),
                raise_exceptions=False,
            )
        except Exception as e:
            print(f"      ❌ country_id={cid} 생성 중 오류: {e}")
            continue

        sub_df = testset.to_pandas()
        sub_df["country_id"] = cid  # ⭐ 어느 국가 질문인지 태깅
        all_dfs.append(sub_df)
        print(f"      ✅ {len(sub_df)}개 질문 생성 (country_id={cid})")

    if not all_dfs:
        print("\n❌ 생성된 질문이 없습니다.")
        return

    # ── 합쳐서 CSV 저장 ──
    print(f"\n💾 [4/4] 전체 결과를 CSV에 저장합니다: {OUTPUT_CSV_PATH}")
    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"   🎉 총 {len(final_df)}개 질문 저장 완료 (국가 {len(all_dfs)}개)")
    print("\n🎉 데이터셋 생성 작업이 성공적으로 종료되었습니다.")
    print("미리보기 (상위 3개):")
    print(final_df.head(3))


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(generate_dataset())
