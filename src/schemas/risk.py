from pydantic import BaseModel, Field
from typing import List, Optional

class RiskRequest(BaseModel):
    country_id: int = Field(..., description="국가 ID")
    travel_purpose: str = Field(..., description="체류 목적")
    visa_type: str = Field(..., description="비자 타입")
    age_band: str = Field(..., description="연령대")

class LawRefResponse(BaseModel):
    law_id: Optional[int] = None
    law_type: str = ""
    article_no: str = ""

class IssueRefResponse(BaseModel):
    issue_id: Optional[int] = None
    title: str = ""
    url: str = ""
    published_date: str = ""

class RiskListResponse(BaseModel):
    risk_title: str
    risk_level: str
    risk_content: str
    risk_actions: List[str]
    law_refs: List[LawRefResponse]
    issue_refs: List[IssueRefResponse] = []

class RiskResult(BaseModel):
    country_id: int
    overall_risk_level: str
    risk_list: List[RiskListResponse]
