"""RAGAS 평가 스크립트 공용 헬퍼 (ragas_evaluate_rag / ragas_evaluate_baseline)."""

import pandas as pd


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
