import asyncio
import json
import os
import sys
import glob

# ------------------------------------------------------------------------------
# 1. 모듈 경로 설정
# ------------------------------------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Vertex AI 관련 import 제거됨 (필요 없음)

from src.core.config import settings
from src.core.models import Law

# ------------------------------------------------------------------------------
# 2. 전역 설정
# ------------------------------------------------------------------------------
DATABASE_URL = settings.ASYNC_DATABASE_URL
# 수동으로 국가/주 ID 지정
TARGET_COUNTRY_ID = 2


async def step2_load_to_db():
    # --------------------------------------------------
    # 3. 초기화 (DB Only) - Vertex AI 제거됨
    # --------------------------------------------------
    print(f"🔧 설정 로드 완료: DB={settings.DB_NAME}")
    print("🔌 DB 연결 중...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # --------------------------------------------------
    # 4. 파일 탐색 (Embedded 파일 찾기)
    # --------------------------------------------------
    # 프로젝트 루트 경로 (src/scripts/load_to_db.py 기준 ../../)
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # 루트 디렉토리에서 '_embedded.jsonl'로 끝나는 모든 파일 찾기
    search_pattern = os.path.join(base_dir, "*_embedded.jsonl")
    files = glob.glob(search_pattern)

    if not files:
        print(
            "🚨 '_embedded.jsonl' 파일을 찾을 수 없습니다. (검색 경로: {})".format(
                search_pattern
            )
        )
        return

    print(f"📂 발견된 데이터 파일: {files}")

    # --------------------------------------------------
    # 5. 데이터 적재 루프
    # --------------------------------------------------
    async with async_session() as session:
        for filename in files:
            print(f"📏 개수 세는 중: {filename}...", end="\r")
            with open(filename, "r", encoding="utf-8") as f_cnt:
                total_lines = sum(1 for line in f_cnt if line.strip())
            print(f"\n🚀 DB 적재 시작: {filename}")

            with open(filename, "r", encoding="utf-8") as f:
                batch_objects = []  # DB 저장용 객체 리스트
                current_count = 0

                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)

                    # (B) Law 객체 생성 (로직 유지)
                    # 1단계 파일에서 'embedding' 값을 가져옵니다.
                    law_obj = Law(
                        country_id=TARGET_COUNTRY_ID,
                        law_type=row.get("law_type"),
                        section_title=row.get("section_title"),
                        article_no=row.get("article_no"),
                        content=row.get("content"),
                        source_url=row.get("source_url"),
                        embedding=row.get("embedding"),  # ★ 여기가 핵심 변경 포인트
                    )
                    batch_objects.append(law_obj)

                    # (C) 배치 처리 (1000개씩 빠르게 저장)
                    # API 호출이 없으므로 배치 사이즈를 늘려도 됩니다.
                    BATCH_SIZE = 1000
                    if len(batch_objects) >= BATCH_SIZE:
                        session.add_all(batch_objects)
                        await session.commit()

                        current_count += len(batch_objects)
                        progress = (current_count / total_lines) * 100
                        print(
                            f"   💾 DB 저장: {current_count}/{total_lines}건 ({progress:.1f}%)",
                            end="\r",
                        )

                        batch_objects = []

                # (D) 남은 데이터 처리
                if batch_objects:
                    session.add_all(batch_objects)
                    await session.commit()
                    current_count += len(batch_objects)
                    print(f"   💾 DB 저장: {current_count}/{total_lines}건 (100.0%)")

            print(f"✅ 파일 완료: {filename} (총 {current_count}건 저장)")

    print("\n🎉 2단계 완료! 모든 데이터가 DB에 안전하게 저장되었습니다.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(step2_load_to_db())
