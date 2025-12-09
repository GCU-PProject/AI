# src/schemas/response.py
from pydantic import BaseModel
from typing import Optional, Any, List


# 개별 국가 검색 결과 요약
class CountryResult(BaseModel):
    related_law_ids: List[int]  # 근거 법률 ID 목록
    summary: str  # 해당 국가 법률 요약


# 비교 분석 결과 (공통점/차이점)
class CompareAnalysis(BaseModel):
    common: str  # 공통점
    diff: str  # 차이점


# 최종 비교 응답 데이터 구조
class CompareResponseResult(BaseModel):
    country_1_result: CountryResult
    country_2_result: CountryResult
    compare_summary: CompareAnalysis


# RAG 서비스의 성공적인 결과 데이터 구조
class ChatResult(BaseModel):
    answer: str  # Gemini LLM이 생성한 최종 답변 텍스트
    related_law_id_list: List[int]  # 답변에 사용된 법률 ID 목록 (Long 대신 Int 가정)
    search_success: bool  # DB 검색 성공 여부 (Threshold 적용 시 유용)
    # distance: Optional[float] = None # 디버깅용으로 최상위 거리 값 포함 가능


# 모든 API의 표준 응답 구조 (isSuccess, code, message)
class CommonResponse(BaseModel):
    isSuccess: bool
    code: str  # 예: "AI200", "COMMON400"
    message: str
    result: Optional[Any] = None  # ChatResult, 에러 내용 등 다양한 데이터가 들어감
