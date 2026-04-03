# src/schemas/compare.py
"""
법률 비교 API의 요청/응답 스키마

비교 결과는 중첩 구조로 되어 있습니다:
CompareResult
├── country_1_result (CountryResult): 기준 국가의 검색 결과
│   ├── related_law_ids: 사용된 법률 ID 목록
│   └── summary: 해당 국가 법률 요약
├── country_2_result (CountryResult): 비교 국가의 검색 결과
│   ├── related_law_ids: 사용된 법률 ID 목록
│   └── summary: 해당 국가 법률 요약
└── compare_summary (CompareAnalysis): 비교 분석 결과
    ├── common: 공통점
    └── diff: 차이점

[왜 중첩 구조인가?]
프론트엔드에서 국가별 요약과 비교 분석을 구분하여 각각 다른 UI 영역에
표시할 수 있도록 데이터를 구조화했습니다.
"""

from pydantic import BaseModel, field_validator
from typing import List


# ---------------------------------------------------
# [요청] Request
# ---------------------------------------------------
class CompareRequest(BaseModel):
    query: str  # 비교할 주제 (예: "음주운전 처벌 비교해줘")
    country_id_1: int  # 기준 국가 ID (예: 1)
    country_id_2: int  # 비교 국가 ID (예: 2)

    @field_validator("query")
    def check_query(cls, v):
        if not v or not v.strip():
            raise ValueError("질문을 입력해주세요.")
        return v


# ---------------------------------------------------
# [응답] Response - 중첩 구조
# ---------------------------------------------------


# 각 국가별 검색 결과
class CountryResult(BaseModel):
    related_law_ids: List[int]  # 근거 법률 ID 목록 (예: [68722, 68724])
    summary: str  # 해당 국가의 법률 내용 요약 (AI 생성)


# 두 국가의 비교 분석 결과
class CompareAnalysis(BaseModel):
    common: str  # 공통점 (AI 생성)
    diff: str  # 차이점 (AI 생성)


# 최종 비교 결과 (CommonResponse의 result에 들어가는 데이터)
class CompareResult(BaseModel):
    country_1_result: CountryResult  # 기준 국가 검색 결과 + 요약
    country_2_result: CountryResult  # 비교 국가 검색 결과 + 요약
    compare_summary: CompareAnalysis  # 공통점/차이점 분석
