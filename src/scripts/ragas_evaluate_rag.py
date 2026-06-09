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
    └── latency_history.csv             ← 실제 응답시간 이력 (한 줄씩 추가)
"""

import argparse
import ast
import asyncio
import os
import sys
import time
from datetime import datetime

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.observability import setup_langsmith

setup_langsmith()

from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.core.llm import embeddings  # 자체 호스팅 Qwen3 (RemoteEmbeddings)
from src.scripts.eval_latency import LatencyRecord, append_latency_history
from src.services.chat_service import (
    CHAT_PROMPT,
    format_docs,
    llm,
    retrieve_laws,
    translate_query,
)

# GCP 인증 환경변수 주입 (config.py의 자동 주입 로직과 동일)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)
from ragas.run_config import RunConfig

# =========================================================
# 1. 설정값
# =========================================================
# 평가셋 경로. _manual/_small 평가셋을 쓰면 결과 파일/이력 실험명에도 자동으로 접미사가 붙는다.
# 전체 평가 시: "data/ragas_testset.csv"
# 발표용 수동 평가셋: "data/ragas_testset_manual.csv"
# 축소 평가 시: "data/ragas_testset_small.csv"
TESTSET_CSV_PATH = "data/ragas_testset.csv"

# 평가셋 종류 → 결과 파일명/이력 실험명에 붙일 접미사
_TESTSET_BASENAME = os.path.basename(TESTSET_CSV_PATH)
if "manual" in _TESTSET_BASENAME:
    NAME_SUFFIX = "_manual"
elif "small" in _TESTSET_BASENAME:
    NAME_SUFFIX = "_small"
else:
    NAME_SUFFIX = ""

EVAL_RESULTS_DIR = "data/eval_results"
# 이력 로그도 small/full을 분리해 섞이지 않게 한다.
EVAL_HISTORY_PATH = os.path.join(EVAL_RESULTS_DIR, f"eval_history{NAME_SUFFIX}.csv")

# [채점 모델] 모든 실험에서 항상 gemini-3-flash-preview로 고정한다.
#   (리걸벤치 기준 법률 분야 정확도가 더 높다고 판단해 채점 기준으로 채택)
#   (이유는 아래 eval_llm 설명 참고)
EVAL_LLM_MODEL = "gemini-3-flash-preview"


def find_metric_column(df: pd.DataFrame, metric_name: str) -> str:
    """RAGAS 버전에 따라 metric_name 또는 metric_name(...) 형태로 저장된 컬럼을 찾는다."""
    candidates = [
        col
        for col in df.columns
        if col == metric_name or col.startswith(f"{metric_name}(")
    ]
    if not candidates:
        raise KeyError(
            f"RAGAS 결과에서 '{metric_name}' 컬럼을 찾지 못했습니다. "
            f"실제 컬럼: {list(df.columns)}"
        )
    return candidates[0]


# =========================================================
# 2. AI 모델 초기화
# =========================================================
# 답변 생성용 LLM, 번역, 검색, 프롬프트 포맷은 실제 서비스(chat_service.py)를
# source of truth로 재사용한다. 서비스 로직이 바뀌면 평가도 같은 로직을 따른다.
# 채점용 임베딩은 검색과 동일한 자체 호스팅 Qwen3(RemoteEmbeddings)로 통일한다.

# 평가 채점용 LLM — 생성 모델(.env의 GCP_MODEL_NAME)과 분리하여 항상 gemini-3-flash-preview로 고정한다.
# [근거]
#  1) 비교 가능성: 채점 모델이 실험마다 바뀌면 점수 변화가 RAG/생성 모델 개선 때문인지
#     채점자 변경 때문인지 구분 불가. → .env의 GCP_MODEL_NAME을 baseline/3.5-flash/
#     3-flash-preview 등으로 바꿔가며 비교 실험을 하더라도 채점자만큼은 절대 바뀌지
#     않아야 이번 라운드에서 만든 모든 결과를 같은 잣대로 비교할 수 있다.
#     ⚠️ 채점자를 gemini-3.5-flash → gemini-3-flash-preview로 교체했으므로,
#     이전 라운드 결과(baseline_no_rag=0.1171, qwen3=0.2993)는 다른 잣대로 잰
#     수치다. 이번에 baseline부터 다시 측정해 새 라운드 안에서만 비교할 것.
#  2) 채점자로 gemini-3-flash-preview를 선택한 이유: 리걸벤치 등에서 법률 분야
#     정확도가 더 높다고 알려진 모델이므로, "법률적으로 옳은지"를 판정하는
#     채점자 역할에는 생성 모델보다 이쪽이 더 적합하다고 판단했다.
#  3) RAGAS 공식 예제도 생성·채점에 동일 모델 사용(gpt-4o)을 권장하지만,
#     "여러 생성 모델을 비교"하는 본 실험에서는 그 원칙을 한 단계 확장해
#     "채점자를 전 실험 공통 기준으로 고정"하는 쪽이 비교 목적에 더 부합한다.
#     공식 문서: "You may choose any model as evaluator LLM for evaluation."
#     (https://docs.ragas.io/en/stable/getstarted/rag_eval/)
#  ※ 목적은 절대 점수가 아닌 A/B 상대 비교이므로 self-evaluation bias는 비교에 영향 없음.
eval_llm = ChatGoogleGenerativeAI(
    model=EVAL_LLM_MODEL,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
    temperature=0,
    thinking_budget=0,
    # thinking 완전 비활성화: claim 분해/NLI는 단순 추출 작업이라 추론 불필요.
    # thinking_level="low"도 호출당 60K+ thinking 토큰 소비 → thinking_budget=0으로 차단.
)


@traceable
async def generate_rag_answer(
    query: str, country_id: int, db
) -> tuple[str, list[str], LatencyRecord]:
    """
    하나의 질문에 대해 실제 RAG 파이프라인을 실행하여
    (답변, 검색된 컨텍스트 리스트, 단계별 응답시간)를 반환합니다.

    country_id: 이 질문이 생성된 국가. 실제 서비스처럼 해당 국가 법률만 검색한다.
    """
    total_started_at = time.perf_counter()

    # Step 1: 번역
    started_at = time.perf_counter()
    translated_query = await translate_query(query)
    translation_time = time.perf_counter() - started_at

    # Step 2: 벡터 검색
    started_at = time.perf_counter()
    docs, _law_ids = await retrieve_laws(translated_query, country_id, db)
    retrieval_time = time.perf_counter() - started_at
    contexts_text = [doc.page_content for doc in docs]

    # Step 3: 검색 결과 없으면 기본 답변
    if not docs:
        total_time = time.perf_counter() - total_started_at
        return (
            "죄송합니다. 질문하신 내용과 관련된 정확한 법률 정보를 찾을 수 없습니다.",
            [],
            {
                "translation": translation_time,
                "retrieval": retrieval_time,
                "generation": 0.0,
                "total": total_time,
                "success": True,
            },
        )

    # Step 4: 답변 생성
    context = format_docs(docs)
    chain = CHAT_PROMPT | llm | StrOutputParser()
    started_at = time.perf_counter()
    answer = await chain.ainvoke({"context": context, "question": query})
    generation_time = time.perf_counter() - started_at
    total_time = time.perf_counter() - total_started_at

    return (
        answer,
        contexts_text,
        {
            "translation": translation_time,
            "retrieval": retrieval_time,
            "generation": generation_time,
            "total": total_time,
            "success": True,
        },
    )


# =========================================================
# 5. 메인 평가 함수
# =========================================================
async def run_evaluation(experiment_name: str, limit: int | None = None):
    """
    전체 평가 파이프라인을 실행합니다.
    1. 테스트셋 로드
    2. 각 질문에 대해 RAG 답변 생성
    3. Ragas로 채점
    4. 결과 저장
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit은 1 이상의 정수여야 합니다.")

    print("=" * 60)
    print(f"🚀 RAGAS 평가 시작 - 실험명: {experiment_name}")
    print("=" * 60)

    # ----- Step 1: 테스트셋 로드 -----
    print("\n📂 [1/4] 테스트셋을 로드합니다...")
    df = pd.read_csv(TESTSET_CSV_PATH)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"   ✅ {len(df)}개 문항 로드 완료")

    # ----- Step 2: RAG 답변 생성 -----
    print("\n🤖 [2/4] 각 질문에 대해 RAG 답변을 생성합니다...")
    responses = []
    retrieved_contexts_list = []
    latency_records: list[LatencyRecord] = []

    # 평가셋은 ragas_generate_dataset.py에서 country_id를 항상 포함해 생성된다.
    # (국가별로 생성 후 태깅) → country_id 컬럼이 없으면 잘못된 데이터이므로 즉시 에러로 알린다.
    if "country_id" not in df.columns:
        raise ValueError(
            "테스트셋에 country_id 컬럼이 없습니다. "
            "ragas_generate_dataset.py로 다시 생성하세요."
        )

    async with AsyncSessionLocal() as db:
        for idx, row in df.iterrows():
            question = row["user_input"]
            country_id = int(row["country_id"])
            print(
                f"   [{idx + 1}/{len(df)}] (country_id={country_id}) {question[:45]}..."
            )

            request_started_at = time.perf_counter()
            try:
                answer, contexts, latency = await generate_rag_answer(
                    question, country_id, db
                )
                responses.append(answer)
                retrieved_contexts_list.append(contexts)
                latency_records.append(latency)
            except Exception as e:
                print(f"   ⚠️ 오류 발생 (건너뜀): {e}")
                responses.append("오류로 인해 답변을 생성할 수 없었습니다.")
                retrieved_contexts_list.append([])
                latency_records.append(
                    {
                        "translation": 0.0,
                        "retrieval": 0.0,
                        "generation": 0.0,
                        "total": time.perf_counter() - request_started_at,
                        "success": False,
                    }
                )

    print("\n⏱️ 실제 사용자 응답시간을 저장합니다...")
    append_latency_history(
        experiment_name=f"{experiment_name}{NAME_SUFFIX}",
        evaluation_type="RAG",
        records=latency_records,
    )

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
    run_config = RunConfig(
        max_retries=5,
        max_wait=300,
        timeout=3600,
        max_workers=1,
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            ContextRecall(),
            ContextPrecision(),
            Faithfulness(),
            FactualCorrectness(),
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

    # 문항별 상세 결과는 파일로 저장하지 않고, 평균 계산에만 사용한다.
    result_df = result.to_pandas()

    # 이력 로그 CSV에 한 줄 추가
    # result_df에서 점수 컬럼의 평균을 바로 계산 (가장 단순하고 안전한 방법)
    metric_cols = {
        "context_recall": find_metric_column(result_df, "context_recall"),
        "context_precision": find_metric_column(result_df, "context_precision"),
        "faithfulness": find_metric_column(result_df, "faithfulness"),
        "factual_correctness": find_metric_column(result_df, "factual_correctness"),
    }
    means = result_df[list(metric_cols.values())].mean()

    avg_scores = {
        "실행일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "실험명": f"{experiment_name}{NAME_SUFFIX}",
        "검색재현율(ContextRecall)": round(means[metric_cols["context_recall"]], 4),
        "검색정밀도(ContextPrecision)": round(
            means[metric_cols["context_precision"]], 4
        ),
        "근거충실도(Faithfulness)": round(means[metric_cols["faithfulness"]], 4),
        "사실정확도(FactualCorrectness)": round(
            means[metric_cols["factual_correctness"]], 4
        ),
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
    for key, label in [
        ("context_recall", "검색 재현율"),
        ("context_precision", "검색 정밀도"),
        ("faithfulness", "근거 충실도"),
        ("factual_correctness", "사실 정확도"),
    ]:
        print(f"   📌 {label}: {round(means[metric_cols[key]], 4)}")
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="평가할 최대 문항 수 (예: --limit 1)",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.name, args.limit))
