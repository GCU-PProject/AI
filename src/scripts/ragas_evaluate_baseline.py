# src/scripts/ragas_evaluate_baseline.py
"""
[RAGAS Baseline 평가 스크립트 — RAG 미적용 (순수 LLM)]

RAG(검색)를 적용하지 않고, 순수 LLM(Gemini)이 자체 지식만으로 답변하게 한 뒤
RAGAS로 채점한다. RAG 적용 버전(ragas_evaluate_rag.py)과 비교하여
"RAG 도입이 성능을 얼마나 향상시켰는가"를 정량적으로 보여주기 위한 기준점이다.

[RAG 버전과의 차이]
  - 검색(retrieve_laws) 단계 없음 → 법률 조항을 전혀 참조하지 않음
  - 질문을 그대로 LLM에 주고 답변만 받음
  - 검색이 없으므로 검색 지표(ContextRecall/ContextPrecision/Faithfulness)는 측정하지 않고,
    RAG 적용 전후 공통 비교가 가능한 답변 지표(FactualCorrectness)만 측정한다.
  - 그 외(테스트셋, 채점 모델, 저장 형식)는 RAG 버전과 동일하게 맞춰 공정 비교한다.

[사용법]
    python src/scripts/ragas_evaluate_baseline.py --name baseline_no_rag

[결과 저장]
    data/eval_results/
    ├── eval_history.csv          ← 실행 이력 (RAG 버전과 공유, 한 줄 추가)
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.observability import setup_langsmith

setup_langsmith()

from src.core.config import settings

# GCP 인증 환경변수 주입
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig

# =========================================================
# 1. 설정값 (RAG 버전과 동일하게 맞춤)
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
# 2. 모델
# =========================================================
# [채점 모델] 모든 실험에서 항상 gemini-3-flash-preview로 고정한다.
#   (리걸벤치 기준 법률 분야 정확도가 더 높다고 판단해 채점 기준으로 채택)
#   생성 모델(.env의 GCP_MODEL_NAME)을 baseline/3.5-flash/3-flash-preview 등으로
#   바꿔가며 비교하는 것이 목적이므로, 채점 기준(judge)마저 같이 바뀌면
#   점수 차이가 "생성 모델 차이" 때문인지 "채점자 차이" 때문인지 구분할 수 없게 된다.
#   → 이번 라운드의 모든 실험(baseline 포함)을 이 채점자로 새로 측정해야
#     서로 비교 가능하다. ⚠️ 채점자가 바뀌었으므로 이전 라운드 결과
#     (baseline_no_rag=0.1171, qwen3=0.2993 — gemini-3.5-flash로 채점)와는
#     절대 점수를 직접 비교하지 말 것 (같은 잣대가 아님).
EVAL_LLM_MODEL = "gemini-3-flash-preview"

# 답변 생성용 LLM — 실제 서비스(get_llm())와 동일하게 .env의 GCP_MODEL_NAME을 그대로 따른다.
#   리걸벤치 등에서 거론되는 다른 모델(예: gemini-3-flash-preview)을 테스트하고 싶으면
#   코드를 건드릴 필요 없이 .env의 GCP_MODEL_NAME 값만 바꾸고 다시 실행하면 된다.
#   (그래야 "우리 서비스가 실제로 쓸 모델"을 그대로 평가하는 것이 되어 결과의 의미가 산다.)
llm = ChatGoogleGenerativeAI(
    model=settings.GCP_MODEL_NAME,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
    temperature=0,
    max_tokens=4096,
    top_k=20,
    top_p=0.7,
)

# 채점용 LLM (모든 실험 공통 — 비교 가능성을 위해 고정, 위 EVAL_LLM_MODEL 설명 참고)
# thinking 완전 비활성화: claim 분해/NLI는 단순 추출 작업이라 추론이 불필요하고,
# thinking_level="low"도 호출당 60K+ thinking 토큰을 소비해 비용/시간이 폭발함.
# thinking_budget=0 → thinking 토큰 0 → 호출당 ~1-2K 토큰으로 정상화.
eval_llm = ChatGoogleGenerativeAI(
    model=EVAL_LLM_MODEL,
    project=settings.GCP_PROJECT_ID,
    location=settings.GCP_LOCATION,
    vertexai=True,
    temperature=0,
    thinking_budget=0,
)

# =========================================================
# 3. Baseline 프롬프트 (RAG 없음 — 순수 LLM 지식으로 답변)
# =========================================================
# RAG 버전의 chat.yaml은 "제공된 조항만 근거로" 답하게 하지만,
# baseline은 참조할 조항이 없으므로 LLM이 자체 지식으로 답하도록 한다.
BASELINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 'Global Legal Assistant'입니다. 전 세계 법률에 대한 질문에 "
            "당신이 알고 있는 지식만으로 한국어로 답변하십시오. "
            "외부 자료나 제공된 조항은 없습니다. 모르면 모른다고 답하되, "
            "가능한 한 정확하고 간결하게 답변하십시오.",
        ),
        ("human", "{question}"),
    ]
)


@traceable
async def generate_llm_answer(query: str) -> str:
    """검색 없이 순수 LLM이 질문에 답변한다."""
    chain = BASELINE_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({"question": query})
    return answer.strip() if answer else ""


# =========================================================
# 4. 메인 평가 함수
# =========================================================
async def run_baseline_evaluation(experiment_name: str, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("--limit은 1 이상의 정수여야 합니다.")

    print("=" * 60)
    print(f"🚀 Baseline(RAG 미적용) 평가 시작 - 실험명: {experiment_name}")
    print("=" * 60)

    # ----- Step 1: 테스트셋 로드 (RAG 버전과 동일한 파일) -----
    print("\n📂 [1/4] 테스트셋을 로드합니다...")
    df = pd.read_csv(TESTSET_CSV_PATH)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"   ✅ {len(df)}개 문항 로드 완료")

    # ----- Step 2: 순수 LLM 답변 생성 (검색 없음) -----
    print("\n🤖 [2/4] 각 질문에 순수 LLM(검색 없음)으로 답변을 생성합니다...")
    responses = []
    for idx, row in df.iterrows():
        question = row["user_input"]
        print(f"   [{idx + 1}/{len(df)}] {str(question)[:45]}...")
        try:
            answer = await generate_llm_answer(question)
            responses.append(answer)
        except Exception as e:
            print(f"   ⚠️ 오류 발생 (건너뜀): {e}")
            responses.append("오류로 인해 답변을 생성할 수 없었습니다.")

    # ----- Step 3: RAGAS 채점 (Baseline/RAG 공통 비교 지표만) -----
    print("\n📊 [3/4] RAGAS 평가를 실행합니다 (FactualCorrectness만)...")
    eval_dataset = Dataset.from_dict(
        {
            "user_input": df["user_input"].tolist(),
            "response": responses,
            "reference": df["reference"].tolist(),
        }
    )

    run_config = RunConfig(
        max_retries=5,
        max_wait=300,
        timeout=3600,
        max_workers=1,
    )
    result = evaluate(
        dataset=eval_dataset,
        # 검색이 없으므로 ContextRecall/Faithfulness는 제외하고,
        # RAG 결과와 공통 비교 가능한 FactualCorrectness만 측정한다.
        metrics=[FactualCorrectness()],
        llm=LangchainLLMWrapper(eval_llm),
        run_config=run_config,
        raise_exceptions=False,
    )

    # ----- Step 4: 결과 저장 (RAG 버전과 동일한 형식) -----
    print("\n💾 [4/4] 결과를 저장합니다...")
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    # 문항별 상세 결과는 파일로 저장하지 않고, 평균 계산에만 사용한다.
    result_df = result.to_pandas()

    # 이력 로그 (검색/근거 지표는 RAG 미적용이므로 공란)
    factual_col = find_metric_column(result_df, "factual_correctness")
    means = result_df[[factual_col]].mean()
    avg_scores = {
        "실행일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "실험명": f"{experiment_name}{NAME_SUFFIX}",
        "검색재현율(ContextRecall)": "-",  # RAG 미적용
        "검색정밀도(ContextPrecision)": "-",  # RAG 미적용
        "근거충실도(Faithfulness)": "-",  # RAG 미적용
        "사실정확도(FactualCorrectness)": round(means[factual_col], 4),
    }

    history_df = pd.DataFrame([avg_scores])
    if os.path.exists(EVAL_HISTORY_PATH):
        existing = pd.read_csv(EVAL_HISTORY_PATH)
        history_df = pd.concat([existing, history_df], ignore_index=True)
    history_df.to_csv(EVAL_HISTORY_PATH, index=False, encoding="utf-8-sig")
    print(f"   📋 이력 로그: {EVAL_HISTORY_PATH}")

    # ----- 결과 출력 -----
    print("\n" + "=" * 60)
    print(f"🎉 Baseline 평가 완료! 실험명: {experiment_name}")
    print("=" * 60)
    print(f"   📌 사실 정확도: {round(means[factual_col], 4)}")
    print("   ℹ️ 검색/근거 지표는 RAG 미적용이므로 측정하지 않음")
    print("=" * 60)


# =========================================================
# 5. 엔트리포인트
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS Baseline(RAG 미적용) 평가")
    parser.add_argument(
        "--name",
        type=str,
        default="baseline_no_rag",
        help="실험명 (예: baseline_no_rag)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="평가할 최대 문항 수 (예: --limit 1)",
    )
    args = parser.parse_args()
    asyncio.run(run_baseline_evaluation(args.name, args.limit))
