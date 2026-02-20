# src/schemas/compare.py
from pydantic import BaseModel
from typing import List


# ---------------------------------------------------
# [요청] Request
# ---------------------------------------------------
class CompareRequest(BaseModel):
    query: str  # 비교할 주제 (예: "음주운전 처벌")
    country_id_1: int  # 첫 번째 국가 ID (예: 1)
    country_id_2: int  # 두 번째 국가 ID (예: 2)


# ---------------------------------------------------
# [응답] Response (Result)
# ---------------------------------------------------
# [내부용] 국가별 요약 정보
class CountryResult(BaseModel):
    related_law_ids: List[int]
    summary: str


# [내부용] 비교 분석 텍스트
class CompareAnalysis(BaseModel):
    common: str
    diff: str


# 비교 결과 데이터 (CommonResponse의 result에 들어갈 내용)
class CompareResult(BaseModel):
    country_1_result: CountryResult
    country_2_result: CountryResult
    compare_summary: CompareAnalysis
