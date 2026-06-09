"""Evaluation latency measurement and history helpers."""

import csv
import os
from datetime import datetime
from typing import TypedDict

import pandas as pd


LATENCY_HISTORY_PATH = "data/eval_results/latency_history.csv"
LATENCY_HISTORY_COLUMNS = [
    "실행일시",
    "실험명",
    "평가종류",
    "문항수",
    "성공수",
    "실패수",
    "평균번역시간(초)",
    "평균검색시간(초)",
    "평균답변생성시간(초)",
    "평균총응답시간(초)",
    "P50총응답시간(초)",
    "P95총응답시간(초)",
    "최소총응답시간(초)",
    "최대총응답시간(초)",
]


class LatencyRecord(TypedDict):
    translation: float
    retrieval: float
    generation: float
    total: float
    success: bool


def append_latency_history(
    experiment_name: str,
    evaluation_type: str,
    records: list[LatencyRecord],
) -> None:
    """Append aggregate user-response latency statistics to the history CSV.

    Latency statistics use successful requests only. Request and failure counts
    are still saved when every request fails.
    """
    successful = [record for record in records if record["success"]]
    failed_count = len(records) - len(successful)

    if successful:
        successful_df = pd.DataFrame(successful)
        total_times = successful_df["total"]
        latency_stats = {
            "평균번역시간(초)": round(successful_df["translation"].mean(), 4),
            "평균검색시간(초)": round(successful_df["retrieval"].mean(), 4),
            "평균답변생성시간(초)": round(successful_df["generation"].mean(), 4),
            "평균총응답시간(초)": round(total_times.mean(), 4),
            "P50총응답시간(초)": round(total_times.quantile(0.50), 4),
            "P95총응답시간(초)": round(total_times.quantile(0.95), 4),
            "최소총응답시간(초)": round(total_times.min(), 4),
            "최대총응답시간(초)": round(total_times.max(), 4),
        }
    else:
        latency_stats = {
            "평균번역시간(초)": None,
            "평균검색시간(초)": None,
            "평균답변생성시간(초)": None,
            "평균총응답시간(초)": None,
            "P50총응답시간(초)": None,
            "P95총응답시간(초)": None,
            "최소총응답시간(초)": None,
            "최대총응답시간(초)": None,
        }

    history_row = {
        "실행일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "실험명": experiment_name,
        "평가종류": evaluation_type,
        "문항수": len(records),
        "성공수": len(successful),
        "실패수": failed_count,
        **latency_stats,
    }

    os.makedirs(os.path.dirname(LATENCY_HISTORY_PATH), exist_ok=True)
    needs_header = not os.path.exists(LATENCY_HISTORY_PATH) or os.path.getsize(
        LATENCY_HISTORY_PATH
    ) == 0
    with open(LATENCY_HISTORY_PATH, "a", newline="", encoding="utf-8-sig") as history:
        writer = csv.DictWriter(history, fieldnames=LATENCY_HISTORY_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(history_row)

    if successful:
        print(f"   ⏱️ 평균 총 응답시간: {history_row['평균총응답시간(초)']}초")
        print(
            "   ⏱️ P50 / P95: "
            f"{history_row['P50총응답시간(초)']}초 / "
            f"{history_row['P95총응답시간(초)']}초"
        )
    else:
        print("   ⚠️ 성공한 응답이 없어 시간 통계는 비워서 저장했습니다.")
    print(f"   📋 응답시간 이력: {LATENCY_HISTORY_PATH}")
