import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from src.core.models import Risk
from src.schemas.risk import RiskResult, RiskListResponse, LawRefResponse, IssueRefResponse

logger = logging.getLogger(__name__)

async def get_risk_cards(
    country_id: int, travel_purpose: str, visa_type: str, age_band: str, db: AsyncSession
) -> RiskResult | None:
    stmt = (
        select(Risk)
        .options(selectinload(Risk.risk_items))
        .where(
            Risk.country_id == country_id,
            Risk.travel_purpose == travel_purpose,
            Risk.visa_type == visa_type,
            Risk.age_band == age_band,
        )
    )
    result = await db.execute(stmt)
    risk = result.scalar_one_or_none()

    if not risk:
        return None

    # Sort constraints
    sorted_items = sorted(risk.risk_items, key=lambda x: x.sort_order)

    risk_list_responses = []
    for item in sorted_items:
        law_refs = [LawRefResponse(**r) for r in item.law_refs] if item.law_refs else []
        issue_refs = [IssueRefResponse(**r) for r in item.issue_refs] if item.issue_refs else []
        
        risk_list_responses.append(
            RiskListResponse(
                risk_title=item.risk_title,
                risk_level=item.risk_level,
                risk_content=item.risk_content,
                risk_actions=item.risk_actions,
                law_refs=law_refs,
                issue_refs=issue_refs
            )
        )

    return RiskResult(
        country_id=risk.country_id,
        overall_risk_level=risk.overall_risk_level,
        risk_list=risk_list_responses
    )
