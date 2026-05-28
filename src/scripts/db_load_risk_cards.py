# src/scripts/db_load_risk_cards.py
"""
리스크 카드 DB 적재 스크립트

data_risk_generate_cards.py로 생성된 data/risk_cards.json을 읽어서
risk / risk_list 테이블에 upsert합니다.

[실행 방법]
python -m src.scripts.db_load_risk_cards

[Upsert 전략]
- 동일한 조합(country_id, travel_purpose, visa_type, age_band)이 이미 있으면:
  → 기존 risk_list 전부 삭제 후 재생성 (clean upsert)
- 없으면: 신규 생성
"""

import asyncio
import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.models import Risk, RiskList

# =========================================================
# 설정
# =========================================================
DATABASE_URL = settings.ASYNC_DATABASE_URL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data", "risk_cards.json")


def is_valid_risk_item(item: dict) -> bool:
    required = [
        "country_id",
        "travel_purpose",
        "visa_type",
        "age_band",
        "overall_risk_level",
        "risk_list",
    ]
    return all(item.get(field) for field in required)


async def load_risk_cards():
    # ─── 1. JSON 파일 읽기 ───
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
        print("   경로를 다시 확인해주세요.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 파일 로드 완료: {INPUT_FILE}")
    print(f"📊 총 {len(data)}개 조합")

    # 필수 필드가 없거나 risk_list가 비어있는 조합 건너뛰기
    valid_data = [d for d in data if is_valid_risk_item(d)]
    skipped = len(data) - len(valid_data)
    if skipped:
        print(f"   ⚠️ 카드가 없는 조합 {skipped}개 건너뜀")

    print(f"   적재 대상: {len(valid_data)}개 조합")

    # ─── 2. DB 연결 ───
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # ─── 3. Upsert 루프 ───
    success_count = 0
    error_count = 0

    async with async_session() as session:
        for idx, item in enumerate(valid_data, 1):
            country_id = item["country_id"]
            travel_purpose = item["travel_purpose"]
            visa_type = item["visa_type"]
            age_band = item["age_band"]
            overall_risk_level = item["overall_risk_level"]
            risk_list = item["risk_list"]

            label = f"[{idx}/{len(valid_data)}] {country_id}/{travel_purpose}/{visa_type}/{age_band}"

            try:
                # 3-1. 기존 레코드 확인
                existing_stmt = select(Risk).where(
                    Risk.country_id == country_id,
                    Risk.travel_purpose == travel_purpose,
                    Risk.visa_type == visa_type,
                    Risk.age_band == age_band,
                )
                existing_result = await session.execute(existing_stmt)
                existing_risk = existing_result.scalar_one_or_none()

                if existing_risk:
                    # 3-2a. 기존 있음 → risk_list 삭제 후 재생성
                    await session.execute(
                        delete(RiskList).where(
                            RiskList.risk_id == existing_risk.risk_id
                        )
                    )
                    existing_risk.overall_risk_level = overall_risk_level
                    risk_id = existing_risk.risk_id
                    print(f"  {label} 🔄 기존 레코드 갱신 (risk_id={risk_id})")
                else:
                    # 3-2b. 기존 없음 → 신규 생성
                    new_risk = Risk(
                        country_id=country_id,
                        travel_purpose=travel_purpose,
                        visa_type=visa_type,
                        age_band=age_band,
                        overall_risk_level=overall_risk_level,
                    )
                    session.add(new_risk)
                    await session.flush()  # risk_id 확보
                    risk_id = new_risk.risk_id
                    print(f"  {label} ✨ 신규 생성 (risk_id={risk_id})")

                # 3-3. risk_list 저장
                for card in risk_list:
                    risk_item = RiskList(
                        risk_id=risk_id,
                        sort_order=card.get("sort_order", 0),
                        risk_title=card["risk_title"],
                        risk_level=card["risk_level"],
                        risk_content=card["risk_content"],
                        risk_actions=card.get("risk_actions", []),
                        law_refs=card.get("law_refs", []),
                        issue_refs=card.get("issue_refs", []),
                    )
                    session.add(risk_item)

                await session.commit()
                success_count += 1

            except Exception as e:
                await session.rollback()
                print(f"  {label} ❌ 오류: {e}")
                error_count += 1

    # ─── 4. 결과 요약 ───
    print(f"\n{'=' * 50}")
    print("📊 적재 결과")
    print(f"   성공: {success_count}건")
    print(f"   실패: {error_count}건")
    print(f"   건너뜀: {skipped}건 (카드 없음)")
    print(f"{'=' * 50}")

    await engine.dispose()
    print("\n🎉 DB 적재 완료!")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(load_risk_cards())
