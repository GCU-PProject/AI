import logging
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Law
from src.schemas.law import LawUrlResponse

logger = logging.getLogger(__name__)


async def get_law_urls(
    law_id_list: List[int], db: AsyncSession
) -> List[LawUrlResponse]:
    """
    법률 ID 목록으로 출처 URL 등 참조 정보를 조회합니다.

    - qna 응답의 related_law_id_list를 그대로 받습니다.
    - 입력 순서를 유지해 반환합니다 (답변에 사용된 순서대로).
    - DB에 없는 ID는 결과에서 제외됩니다 (있는 것만 반환).
    """
    if not law_id_list:
        return []

    # 한 번의 쿼리로 모두 조회 후, 입력 순서대로 재정렬한다.
    result = await db.execute(select(Law).where(Law.law_id.in_(law_id_list)))
    laws = {law.law_id: law for law in result.scalars().all()}

    responses: List[LawUrlResponse] = []
    for law_id in law_id_list:
        law = laws.get(law_id)
        if law is None:
            continue
        responses.append(
            LawUrlResponse(
                law_id=law.law_id,
                law_type=law.law_type,
                article_no=law.article_no,
                section_title=law.section_title or "",
                source_url=law.source_url,
            )
        )
    return responses
