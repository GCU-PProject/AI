from typing import List, Optional

from pydantic import BaseModel, Field


class LawUrlRequest(BaseModel):
    # qna 응답의 related_law_id_list를 그대로 넣으면 됩니다.
    law_id_list: List[int] = Field(
        ..., min_length=1, description="법률 ID 목록 (예: [68722, 68724])"
    )


class LawUrlResponse(BaseModel):
    law_id: int
    law_type: str = ""  # 법률 코드 (예: VEH, PEN)
    article_no: str = ""  # 조항 번호 (예: 23152.)
    section_title: str = ""  # 목차/카테고리
    source_url: Optional[str] = None  # 출처 URL (없으면 null)
