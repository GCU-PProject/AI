"""
MTEB 영어 법률(Legal) Retrieval 임베딩 모델 벤치마크 전수 분석
──────────────────────────────────────────────────────────────
[적용 필터]
  1. Domain    : Legal (법률)
  2. Task Type : Retrieval
  3. Language  : eng (영어)
  4. Model     : 필터 없음 (전수 조사)

[대상 태스크] - MTEB에 실제 등록된 영어 법률 Retrieval 태스크만 사용
  • BillSumCA                    : 캐나다 법안 검색
  • BillSumUS                    : 미국 법안 검색
  • BarExamQA                    : 미국 변호사 시험 QA 검색
  • AILACasedocs                 : 판례 문서 검색
  • AILAStatutes                 : 법령 조문 검색
  • GovReport                    : 정부 보고서 검색
  • LegalBenchConsumerContractsQA: 소비자 계약 QA 검색
  • LegalBenchCorporateLobbying  : 기업 로비 법률 검색
  • LegalSummarization           : 법률 문서 요약 검색

[파일 경로 구조]
  results/{조직명}__{모델명}/{커밋해시}/{태스크이름}.json
  예: results/intfloat__multilingual-e5-small/fd1525a9.../BillSumCA.json
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from huggingface_hub import HfApi
from tqdm import tqdm

# ──────────────────────────────────────────────
# [사용자 설정] 여기만 수정하면 됩니다
# ──────────────────────────────────────────────

LEGAL_RETRIEVAL_TASKS = [
    "BillSumCA",
    "BillSumUS",
    "BarExamQA",
    "AILACasedocs",
    "AILAStatutes",
    "GovReport",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
]

MAX_WORKERS = 30
TOP_N_DISPLAY = 30
TIMEOUT_SEC = 5


def fetch_score(file_path: str) -> dict | None:
    """단일 JSON 파일에서 벤치마크 점수를 파싱하여 반환"""
    # 경로 구조: results/{org}__{model}/{hash}/{TaskName}.json
    parts = file_path.split("/")
    if len(parts) < 4:
        return None

    model_name = parts[1].replace("__", "/")  # parts[1]이 모델명
    task_name = parts[-1].replace(".json", "")
    url = f"https://huggingface.co/datasets/mteb/results/resolve/main/{file_path}"

    try:
        resp = requests.get(url, timeout=TIMEOUT_SEC)
        if resp.status_code != 200:
            return None

        scores_dict = resp.json().get("scores", {})
        score = None

        for split in ["test", "train", "validation"]:
            split_data = scores_dict.get(split, [])
            if split_data and isinstance(split_data, list):
                for item in split_data:
                    if "ndcg_at_10" in item:
                        score = item["ndcg_at_10"]
                        break
                if score is not None:
                    break

        if score is not None:
            return {
                "Model": model_name,
                "Task": task_name,
                "Score (NDCG@10)": round(score, 4),
            }
    except Exception:
        pass
    return None


def main():
    # ── 1. 적용 필터 출력 ──
    print("\n" + "=" * 70)
    print("📋 적용 필터")
    print("=" * 70)
    print(f"  Domain     : Legal")
    print(f"  Task Type  : Retrieval")
    print(f"  Language   : eng")
    print(f"  대상 태스크:")
    for t in LEGAL_RETRIEVAL_TASKS:
        print(f"    - {t}")
    print("=" * 70)

    # ── 2. MTEB 파일 목록 스캔 ──
    api = HfApi()
    try:
        print("\n🔄 MTEB 파일 목록 스캔 중...")
        repo_files = api.list_repo_files(repo_id="mteb/results", repo_type="dataset")
        print(f"✅ 총 {len(repo_files):,}개 파일 확인")
    except Exception as e:
        print(f"❌ 실패: {e}")
        return

    # ── 3. 대상 태스크 파일만 필터링 ──
    target_files = []
    for file_path in repo_files:
        if not file_path.endswith(".json"):
            continue
        task_in_file = file_path.split("/")[-1].replace(".json", "")
        if task_in_file in LEGAL_RETRIEVAL_TASKS:
            target_files.append(file_path)

    print(f"🎯 법률 Retrieval 태스크 매칭 파일: {len(target_files):,}개\n")
    if not target_files:
        print("⚠️ 매칭 파일 없음")
        return

    # ── 4. 병렬 다운로드 및 점수 파싱 ──
    print(f"📥 {len(target_files):,}개 파일을 {MAX_WORKERS}개 스레드로 병렬 수집 중...")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_score, fp): fp for fp in target_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="수집 중"):
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        print("\n⚠️ 수집된 데이터 없음\n")
        return

    df = pd.DataFrame(results)

    # ── 5. 모델별 평균 점수 + 참여 태스크 수 산출 ──
    ranking = (
        df.groupby("Model")
        .agg(
            평균점수=("Score (NDCG@10)", "mean"),
            참여태스크수=("Score (NDCG@10)", "count"),
        )
        .reset_index()
    )
    ranking["평균점수"] = ranking["평균점수"].round(4)
    ranking = ranking.sort_values(by="평균점수", ascending=False)

    # ── 6. 결과 출력 ──
    print("\n" + "=" * 70)
    print(f"⚖️ 영어 법률(Legal) Retrieval 전수 분석 리더보드")
    print(f"   총 {len(ranking):,}개 모델 중 상위 {TOP_N_DISPLAY}개 표시")
    print("=" * 70)
    print(ranking.head(TOP_N_DISPLAY).to_markdown(index=False))

    # ── 7. CSV 저장 ──
    csv_path = os.path.join(
        os.path.dirname(__file__), "mteb_legal_retrieval_leaderboard.csv"
    )
    ranking.to_csv(csv_path, index=False)
    print("=" * 70)
    print(f"💾 전체 {len(ranking):,}개 모델 랭킹 저장: {csv_path}")
    print(f"   (터미널에는 상위 {TOP_N_DISPLAY}개만 표시, CSV에는 전체 포함)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
