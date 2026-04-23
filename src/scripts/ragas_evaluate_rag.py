# src/scripts/ragas_evaluate_rag.py
"""
[RAGAS 평가 스크립트]

이 스크립트는 생성된 골든 데이터셋(ragas_testset.csv)의 질문들을
실제 RAG 파이프라인(번역 → 검색 → 답변 생성)에 통과시킨 뒤,
Ragas 0.4.x 프레임워크로 자동 채점하여 시스템 성능을 측정합니다.

[사용법]
    python src/scripts/ragas_evaluate_rag.py --name baseline
    python src/scripts/ragas_evaluate_rag.py --name add_reranker

[결과 저장 구조]
    data/eval_results/
    ├── eval_history.csv                ← 실행 이력 (한 줄씩 추가)
    ├── 20260325_baseline.csv           ← 문항별 상세 점수
    └── 20260328_add_reranker.csv
"""

import sys
import os
import asyncio
import argparse
import ast
from datetime import datetime

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.core.models import Law

# GCP 인증 환경변수 주입 (config.py의 자동 주입 로직과 동일)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sqlalchemy import select

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# =========================================================
# 1. 설정값
# =========================================================
TESTSET_CSV_PATH = "data/ragas_testset.csv"
EVAL_RESULTS_DIR = "data/eval_results"
EVAL_HISTORY_PATH = os.path.join(EVAL_RESULTS_DIR, "eval_history.csv")

# chat_service.py와 동일한 검색 파라미터
TOP_K = settings.RAG_TOP_K
MAX_DISTANCE_THRESHOLD = settings.RAG_MAX_DISTANCE_THRESHOLD

# =========================================================
# 2. AI 모델 초기화 (chat_service.py와 동일)
# =========================================================
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
)

# 답변 생성용 LLM (서비스와 동일한 모델 사용)
llm = ChatVertexAI(
    model_name=settings.GCP_MODEL_NAME,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    temperature=0,
    max_output_tokens=4096,
    top_k=20,
    top_p=0.7,
)

# 평가 채점용 LLM (고성능 Pro 모델 사용)
eval_llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    temperature=0,
)

# =========================================================
# 3. 프롬프트 로드 (chat_service.py와 동일)
# =========================================================
translation_yaml = load_prompt("src/prompts/translation.yaml", encoding="utf-8")
TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", translation_yaml.template),
        ("human", "{query}"),
    ]
)

chat_yaml = load_prompt("src/prompts/chat.yaml", encoding="utf-8")
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", chat_yaml.template),
        ("human", "{question}"),
    ]
)


# =========================================================
# 4. RAG 파이프라인 함수 (chat_service.py 로직 재사용)
# =========================================================
async def translate_query(query: str) -> str:
    """질문을 영어로 번역합니다."""
    chain = TRANSLATION_PROMPT | llm | StrOutputParser()
    translated = await chain.ainvoke({"query": query})
    return translated.strip() if translated else query


async def retrieve_laws(
    query: str, country_id: int, db
) -> tuple[list[Document], list[str]]:
    """벡터 유사도 기반으로 관련 법률 조항을 검색합니다."""
    query_vector = embeddings.embed_query(query)

    stmt = (
        select(Law, Law.embedding.l2_distance(query_vector).label("distance"))
        .where(Law.country_id == country_id)
        .order_by(Law.embedding.l2_distance(query_vector))
        .limit(TOP_K)
    )

    result = await db.execute(stmt)
    rows = result.all()

    documents = []
    contexts_text = []

    for row in rows:
        law = row[0]
        distance = row[1]

        if distance <= MAX_DISTANCE_THRESHOLD:
            doc = Document(
                page_content=law.content,
                metadata={
                    "law_id": law.law_id,
                    "law_type": law.law_type,
                    "section_title": law.section_title or "",
                    "article_no": law.article_no,
                    "distance": distance,
                },
            )
            documents.append(doc)
            contexts_text.append(law.content)

    return documents, contexts_text


def format_docs(docs: list[Document]) -> str:
    """검색된 Document 리스트를 프롬프트용 텍스트로 변환합니다."""
    if not docs:
        return "(관련 법률 정보 없음)"

    formatted = []
    for doc in docs:
        meta = doc.metadata
        formatted.append(
            f"[{meta['law_type']} {meta['article_no']}]\n"
            f"- 목차: {meta['section_title']}\n"
            f"- 내용: {doc.page_content}\n"
            f"--------------------------------------------------"
        )
    return "\n".join(formatted)


async def generate_rag_answer(query: str, db) -> tuple[str, list[str]]:
    """
    하나의 질문에 대해 실제 RAG 파이프라인을 실행하여
    (답변, 검색된 컨텍스트 리스트)를 반환합니다.
    """
    # country_id=1 (캘리포니아)로 고정
    country_id = 1

    # Step 1: 번역
    translated_query = await translate_query(query)

    # Step 2: 벡터 검색
    docs, contexts_text = await retrieve_laws(translated_query, country_id, db)

    # Step 3: 검색 결과 없으면 기본 답변
    if not docs:
        return (
            "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다.",
            [],
        )

    # Step 4: 답변 생성
    context = format_docs(docs)
    chain = CHAT_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": query})

    return answer, contexts_text


# =========================================================
# 5. 메인 평가 함수
# =========================================================
async def run_evaluation(experiment_name: str):
    """
    전체 평가 파이프라인을 실행합니다.
    1. 테스트셋 로드
    2. 각 질문에 대해 RAG 답변 생성
    3. Ragas로 채점
    4. 결과 저장
    """
    print("=" * 60)
    print(f"🚀 RAGAS 평가 시작 - 실험명: {experiment_name}")
    print("=" * 60)

    # ----- Step 1: 테스트셋 로드 -----
    print("\n📂 [1/4] 테스트셋을 로드합니다...")
    df = pd.read_csv(TESTSET_CSV_PATH)
    print(f"   ✅ {len(df)}개 문항 로드 완료")

    # ----- Step 2: RAG 답변 생성 -----
    print("\n🤖 [2/4] 각 질문에 대해 RAG 답변을 생성합니다...")
    responses = []
    retrieved_contexts_list = []

    async with AsyncSessionLocal() as db:
        for idx, row in df.iterrows():
            question = row["user_input"]
            print(f"   [{idx + 1}/{len(df)}] {question[:50]}...")

            try:
                answer, contexts = await generate_rag_answer(question, db)
                responses.append(answer)
                retrieved_contexts_list.append(contexts)
            except Exception as e:
                print(f"   ⚠️ 오류 발생 (건너뜀): {e}")
                responses.append("오류로 인해 답변을 생성할 수 없었습니다.")
                retrieved_contexts_list.append([])

    # ----- Step 3: Ragas 평가용 Dataset 구성 -----
    print("\n📊 [3/4] Ragas 평가를 실행합니다...")

    # reference_contexts 컬럼은 CSV에서 문자열로 저장되므로 리스트로 변환
    reference_contexts = []
    for ctx_str in df["reference_contexts"]:
        try:
            parsed = ast.literal_eval(ctx_str)
            if isinstance(parsed, list):
                reference_contexts.append(parsed)
            else:
                reference_contexts.append([str(parsed)])
        except Exception:
            reference_contexts.append([str(ctx_str)])

    eval_dataset = Dataset.from_dict(
        {
            "user_input": df["user_input"].tolist(),
            "response": responses,
            "reference": df["reference"].tolist(),
            "retrieved_contexts": retrieved_contexts_list,
            "reference_contexts": reference_contexts,
        }
    )

    # 평가 실행
    run_config = RunConfig(max_retries=5, max_wait=120, timeout=300)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            AnswerCorrectness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=LangchainLLMWrapper(eval_llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=run_config,
        raise_exceptions=False,
    )

    # ----- Step 4: 결과 저장 -----
    print("\n💾 [4/4] 결과를 저장합니다...")

    # 결과 디렉토리 생성
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    # 4-1. 상세 결과 CSV 저장 (문항별 점수)
    today = datetime.now().strftime("%Y%m%d")
    detail_filename = f"{today}_{experiment_name}.csv"
    detail_path = os.path.join(EVAL_RESULTS_DIR, detail_filename)

    result_df = result.to_pandas()
    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"   📄 상세 결과: {detail_path}")

    # 4-2. 이력 로그 CSV에 한 줄 추가
    # result_df에서 점수 컬럼의 평균을 바로 계산 (가장 단순하고 안전한 방법)
    metric_cols = [
        "answer_correctness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    means = result_df[metric_cols].mean()

    avg_scores = {
        "실행일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "실험명": experiment_name,
        "정답일치도(AnswerCorrectness)": round(means["answer_correctness"], 4),
        "답변관련성(AnswerRelevancy)": round(means["answer_relevancy"], 4),
        "검색정밀도(ContextPrecision)": round(means["context_precision"], 4),
        "검색재현율(ContextRecall)": round(means["context_recall"], 4),
    }

    history_df = pd.DataFrame([avg_scores])

    if os.path.exists(EVAL_HISTORY_PATH):
        existing = pd.read_csv(EVAL_HISTORY_PATH)
        history_df = pd.concat([existing, history_df], ignore_index=True)

    history_df.to_csv(EVAL_HISTORY_PATH, index=False, encoding="utf-8-sig")
    print(f"   📋 이력 로그: {EVAL_HISTORY_PATH}")

    # ----- 결과 출력 -----
    print("\n" + "=" * 60)
    print(f"🎉 평가 완료! 실험명: {experiment_name}")
    print("=" * 60)
    for col_name, label in zip(
        metric_cols, ["정답 일치도", "답변 관련성", "검색 정밀도", "검색 재현율"]
    ):
        print(f"   📌 {label}: {round(means[col_name], 4)}")
    print("=" * 60)


# =========================================================
# 6. 엔트리포인트
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 평가 스크립트")
    parser.add_argument(
        "--name",
        type=str,
        default="unnamed",
        help="실험명 (예: baseline, add_reranker, prompt_tuning_v2)",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.name))
