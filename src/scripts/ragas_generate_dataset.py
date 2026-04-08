# src/scripts/ragas_generate_dataset.py
"""
[RAGAS 데이터셋 생성 스크립트]

이 스크립트는 DB(PostgreSQL)에 저장된 법률 문서(chunk)들을 불러와,
RAGAS 프레임워크를 이용해 RAG 평가용 합성 데이터셋(Synthetic Dataset)을 자동 생성합니다.

- LLM & Embeddings: Vertex AI (gemini-2.0-flash, text-embedding-005)
- 소스 데이터: `laws` 테이블의 실제 법률 조항
- 목표 데이터셋 크기: 20개 (SingleHop 60%, MultiHop 40%)

[RAGAS 0.4.x API 기준]
- 구버전(0.1.x)의 evolutions(simple, reasoning, multi_context, conditional)이
  신버전(0.4.x)의 synthesizers(SingleHop, MultiHopAbstract, MultiHopSpecific)로 변경됨
- InMemoryDocumentStore → KnowledgeGraph (내부 자동 생성)
- language="korean" 파라미터 → llm_context로 한국어 생성 유도
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

# ====================================================================
# RAGAS 관련 임포트 (0.4.x 신버전 API)
# ====================================================================
# TestsetGenerator: 테스트셋 생성의 메인 클래스
# - 구버전: from ragas.testset.generator import TestsetGenerator
# - 신버전: from ragas.testset import TestsetGenerator (경로가 바뀜)
from ragas.testset import TestsetGenerator

# Synthesizers: 질문 유형(난이도)별 생성기
# - 구버전의 evolutions(simple, reasoning, multi_context, conditional)을 대체
# - SingleHopSpecific: 하나의 chunk로 답할 수 있는 단순 질문 (≈ 구버전 simple)
# - MultiHopAbstract: 여러 chunk를 종합해 추상적 개념을 추론 (≈ 구버전 reasoning)
# - MultiHopSpecific: 여러 chunk에서 구체적 정보를 연결 (≈ 구버전 multi_context)
# ※ 구버전의 conditional(조건부 질문)은 신버전에 대응 타입이 없어 MultiHopSpecific에 통합
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

# LangchainLLMWrapper: LangChain LLM 객체를 RAGAS 내부 형식으로 감싸는 래퍼
# - TestsetGenerator.from_langchain()은 내부에서 자동으로 래핑해줌
# - 하지만 query_distribution을 수동으로 만들 때는 직접 래핑해서 synthesizer에 넘겨야 함
from ragas.llms import LangchainLLMWrapper

from src.core.database import AsyncSessionLocal
from src.core.models import Law
from src.core.config import settings
from ragas.run_config import RunConfig

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
TESTSET_SIZE = 20
DOCUMENT_LIMIT = 50  # 테스트셋 생성의 재료가 될 문서 개수 (너무 많으면 느림)
OUTPUT_CSV_PATH = "data/ragas_testset.csv"

# llm_context: 구버전의 language="korean" 파라미터를 대체
# - 0.4.x에서는 language 파라미터가 제거됨
# - 대신 llm_context 문자열을 LLM에게 전달하여 한국어 생성을 유도
# - TestsetGenerator 생성 시 전달하면, 내부의 모든 synthesizer에 자동 전파됨
LLM_CONTEXT = (
    "You are evaluating an Australian legal AI assistant for Korean-speaking users. "
    "CRITICAL MANDATORY RULE: YOU MUST GENERATE ALL QUESTIONS AND ANSWERS EXCLUSIVELY IN THE KOREAN LANGUAGE (한국어). "
    "Even if instructed to adopt an English-speaking persona or a specific English style (e.g. 'POOR_GRAMMAR'), YOU MUST TRANSLATE your final thought into highly natural Korean before outputting. NEVER use English for the generated questions or answers. "
    'IMPORTANT: 1) ALWAYS output valid JSON. 2) Ensure internal quotes inside JSON strings are properly escaped (e.g. \\" instead of "). '
    "3) Do not include markdown code block backticks around your JSON payload."
)


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
            # LangChain Document 형식으로 변환
            # - generate_with_chunks()는 Document 또는 단순 문자열(str)을 모두 받을 수 있음
            # - metadata를 포함시키면 생성된 질문에 출처 정보가 남아서 추적에 유용
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

    # ── Vertex AI 모델 초기화 ──
    llm = ChatVertexAI(
        model_name="gemini-2.5-pro",  # pro 모델로 고정
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        temperature=0.0,  # 법률 AI의 일관성과 정확성을 위해 0.0으로 고정
    )

    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )

    # ── TestsetGenerator 생성 (신버전 API) ──
    # 구버전: TestsetGenerator.from_langchain(generator_llm=, critic_llm=, embeddings=, docstore=)
    # 신버전: TestsetGenerator.from_langchain(llm=, embedding_model=, llm_context=)
    #
    # from_langchain()이 내부에서 자동으로:
    #   1. llm → LangchainLLMWrapper(llm) 으로 래핑
    #   2. embedding_model → LangchainEmbeddingsWrapper(embedding_model) 으로 래핑
    # → 따라서 수동 래핑이 필요 없음!
    #
    # llm_context: 여기서 전달하면 generate() 호출 시 default_query_distribution()에 자동 전파
    print("🛠️ [3/5] Testset Generator를 설정합니다...")
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
    )

    # ── query_distribution 설정 (질문 난이도 비율) ──
    # 구버전: distributions = {simple: 0.6, reasoning: 0.2, multi_context: 0.1, conditional: 0.1}
    # 신버전: query_distribution = [(SynthesizerInstance, 비율), ...]
    #
    # ※ query_distribution을 None으로 두면 default_query_distribution()이 호출되어
    #    3가지 synthesizer가 균등 분배(각 33.3%)됨.
    #    우리 서비스는 비전문가(유학생/여행자) 대상이므로 쉬운 질문 비율을 높이기 위해 수동 설정.
    #
    # ※ synthesizer 인스턴스에 llm을 넘기려면 RAGAS 내부 형식(LangchainLLMWrapper)이 필요
    #    → from_langchain()은 generator 내부만 래핑하므로, synthesizer용으로는 직접 래핑
    ragas_llm = LangchainLLMWrapper(llm)

    query_distribution = [
        # SingleHopSpecific (≈ simple): 50%
        # 하나의 법률 chunk에서 바로 답할 수 있는 직접적인 질문
        # 예: "호주에서 불법 해고의 정의는 무엇인가요?"
        (
            SingleHopSpecificQuerySynthesizer(llm=ragas_llm, llm_context=LLM_CONTEXT),
            0.5,
        ),
        # MultiHopAbstract (≈ reasoning): 20%
        # 여러 chunk를 종합하여 추상적 개념을 추론해야 하는 질문
        # 예: "고용주의 의무를 종합하면 어떤 원칙이 적용되나요?"
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm, llm_context=LLM_CONTEXT), 0.3),
        # MultiHopSpecific (≈ multi_context + conditional): 30%
        # 여러 chunk에서 구체적 정보를 연결해야 하는 질문
        # 예: "Section 385의 예외 조항이 Section 386에도 적용되나요?"
        # ※ 구버전 conditional(0.1) + multi_context(0.1)을 통합
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm, llm_context=LLM_CONTEXT), 0.2),
    ]

    print(
        f"⚙️ [4/5] RAGAS 테스트셋 생성 시작! (목표: {TESTSET_SIZE}개, 시간이 걸릴 수 있습니다)"
    )
    print(f"   - 비율: SingleHop 50%, MultiHopAbstract 20%, MultiHopSpecific 30%")

    try:
        # ── 테스트셋 생성 (신버전 API) ──
        # 구버전: generate_with_langchain_docs(documents=, test_size=, distributions=, language=)
        # 신버전: generate_with_chunks(chunks=, testset_size=, query_distribution=)
        #
        # generate_with_chunks()를 쓰는 이유:
        #   우리 DB의 법률 문서는 이미 조항(article) 단위로 쪼개진 chunk임.
        #   generate_with_langchain_docs()는 내부에서 문서를 다시 분할(split)하지만,
        #   generate_with_chunks()는 NodeType.CHUNK로 직접 처리하여 원본 조문을 보존함.
        #
        # 내부 동작 순서:
        #   1. chunks → KnowledgeGraph 노드(NodeType.CHUNK)로 변환
        #   2. default_transforms_for_prechunked() 적용 (요약, NER, 임베딩 추출 등)
        #   3. query_distribution에 따라 시나리오 생성 → 질문/답변 합성
        testset = generator.generate_with_chunks(
            chunks=docs,  # LangChain Document 리스트
            testset_size=TESTSET_SIZE,  # 생성할 테스트 샘플 수 (구버전: test_size)
            query_distribution=query_distribution,  # 질문 유형별 비율 (구버전: distributions)
            with_debugging_logs=True,  # 디버깅 로그 활성화
            run_config=RunConfig(
                timeout=120, max_retries=10, max_workers=4
            ),  # 재시도 10회 등 여유 부여
            raise_exceptions=False,  # JSON 파싱 실패 같은 에러가 나도 하나만 건너뛰고 전체 코드가 뻗지 않도록 방지
        )
    except Exception as e:
        print(f"\n❌ 데이터셋 생성 중 오류 발생: {e}")
        return

    # ── CSV 저장 ──
    # to_pandas() 결과 컬럼 (구버전 → 신버전):
    #   question → user_input           : 생성된 질문
    #   contexts → reference_contexts   : 질문의 근거가 된 원본 chunk들
    #   ground_truth → reference        : LLM이 생성한 정답 (ground truth)
    #   (신규) synthesizer_name          : 어떤 synthesizer가 만든 질문인지 (난이도 추적용)
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
