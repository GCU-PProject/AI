"""countries 테이블에 초기 데이터 삽입"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

from src.core.config import settings

COUNTRIES = [
    {
        "country_id": 1,
        "country_code": "US",
        "state_code": None,
        "country_name": "United States",
        "state_name": None,
    },
    {
        "country_id": 2,
        "country_code": "US",
        "state_code": "CA",
        "country_name": "United States",
        "state_name": "California",
    },
    {
        "country_id": 3,
        "country_code": "US",
        "state_code": "NY",
        "country_name": "United States",
        "state_name": "New York",
    },
    {
        "country_id": 4,
        "country_code": "CA",
        "state_code": None,
        "country_name": "Canada",
        "state_name": None,
    },
    {
        "country_id": 5,
        "country_code": "CA",
        "state_code": "ON",
        "country_name": "Canada",
        "state_name": "Ontario",
    },
    {
        "country_id": 6,
        "country_code": "CA",
        "state_code": "BC",
        "country_name": "Canada",
        "state_name": "British Columbia",
    },
]


def insert_countries() -> None:
    engine = create_engine(settings.DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO countries (
                    country_id,
                    country_code,
                    state_code,
                    country_name,
                    state_name
                )
                VALUES (
                    :country_id,
                    :country_code,
                    :state_code,
                    :country_name,
                    :state_name
                )
                ON CONFLICT (country_id) DO UPDATE
                SET
                    country_code = EXCLUDED.country_code,
                    state_code = EXCLUDED.state_code,
                    country_name = EXCLUDED.country_name,
                    state_name = EXCLUDED.state_name;
                """
            ),
            COUNTRIES,
        )
        conn.execute(
            text("SELECT setval(pg_get_serial_sequence('countries', 'country_id'), 6, true);")
        )
        rows = conn.execute(
            text(
                """
                SELECT country_id, country_code, state_code, country_name, state_name
                FROM countries
                ORDER BY country_id;
                """
            )
        ).all()

    print("✅ Countries 데이터 삽입 성공!")
    for row in rows:
        print(f"  {row}")


if __name__ == "__main__":
    insert_countries()
