import asyncio
import argparse
import glob
import json
import os
import sys
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. 모듈 경로 설정
# ------------------------------------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.models import Law

DATABASE_URL = settings.ASYNC_DATABASE_URL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATTERN = os.path.join(BASE_DIR, "data", "*_Embedded.jsonl")
BATCH_SIZE = 1000
REQUIRED_FIELDS = (
    "country_id",
    "law_type",
    "section_title",
    "article_no",
    "content",
    "source_url",
    "embedding",
)


def parse_datetime(value: str | None) -> datetime | None:
    """JSONL 문자열 날짜를 PostgreSQL DateTime에 넣을 수 있는 값으로 변환합니다."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def find_input_files() -> list[str]:
    return sorted(glob.glob(DEFAULT_PATTERN))


def build_law(row: dict) -> Law | None:
    if any(row.get(field) is None for field in REQUIRED_FIELDS):
        return None

    return Law(
        country_id=row["country_id"],
        law_type=row["law_type"],
        section_title=row["section_title"],
        article_no=row["article_no"],
        content=row["content"],
        source_url=row["source_url"],
        enactment_date=parse_datetime(row.get("enactment_date")),
        amendment_date=parse_datetime(row.get("amendment_date")),
        embedding=row["embedding"],
    )


async def load_law_data(
    limit: int | None = None,
) -> None:
    # --------------------------------------------------
    # 3. 초기화 (DB Only) - Vertex AI 제거됨
    # --------------------------------------------------
    print(f"🔧 설정 로드 완료: DB={settings.DB_NAME}")
    print("🔌 DB 연결 중...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    files = find_input_files()

    if not files:
        print(f"🚨 Embedded JSONL 파일을 찾을 수 없습니다. (검색 경로: {DEFAULT_PATTERN})")
        await engine.dispose()
        return

    print(f"📂 발견된 데이터 파일: {files}")

    try:
        async with async_session() as session:
            for filename in files:
                print(f"📏 개수 세는 중: {filename}...", end="\r")
                with open(filename, "r", encoding="utf-8") as f_cnt:
                    total_lines = sum(1 for line in f_cnt if line.strip())
                if limit:
                    total_lines = min(total_lines, limit)
                print(f"\n🚀 DB 적재 시작: {filename}")

                batch_objects = []
                saved_count = 0
                skipped_count = 0

                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        if limit and saved_count + skipped_count >= limit:
                            break
                        if not line.strip():
                            continue

                        row = json.loads(line)
                        law_obj = build_law(row)
                        if law_obj is None:
                            skipped_count += 1
                            continue

                        batch_objects.append(law_obj)

                        if len(batch_objects) >= BATCH_SIZE:
                            session.add_all(batch_objects)
                            await session.commit()

                            saved_count += len(batch_objects)
                            progress = (saved_count + skipped_count) / total_lines * 100
                            print(
                                f"   💾 DB 저장: {saved_count}/{total_lines}건 "
                                f"(스킵 {skipped_count}건, {progress:.1f}%)",
                                end="\r",
                            )
                            batch_objects = []

                    if batch_objects:
                        session.add_all(batch_objects)
                        await session.commit()
                        saved_count += len(batch_objects)

                print(
                    f"\n✅ 파일 완료: {filename} "
                    f"(저장 {saved_count}건, 스킵 {skipped_count}건)"
                )

        print("\n🎉 법률 데이터 DB 적재 완료!")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 완료 법률 JSONL DB 적재")
    parser.add_argument("--limit", type=int, default=None, help="테스트용 최대 적재 행 수")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(load_law_data(limit=args.limit))
