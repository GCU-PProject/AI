"""countries 테이블에 초기 데이터 삽입"""

import psycopg2
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=os.getenv("DB_PORT", "5432"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_NAME"),
    sslmode="require",
)
cur = conn.cursor()

cur.execute(
    """
    INSERT INTO countries (country_id, country_code, state_code, country_name, state_name)
    VALUES 
        (1, 'US', NULL, 'United States', NULL),
        (2, 'US', 'CA', 'United States', 'California'),
        (3, 'US', 'NY', 'United States', 'New York'),
        (4, 'CA', NULL, 'Canada', NULL),
        (5, 'CA', 'ON', 'Canada', 'Ontario'),
        (6, 'CA', 'BC', 'Canada', 'British Columbia')
    ON CONFLICT (country_id) DO UPDATE
    SET
        country_code = EXCLUDED.country_code,
        state_code = EXCLUDED.state_code,
        country_name = EXCLUDED.country_name,
        state_name = EXCLUDED.state_name;
"""
)

cur.execute(
    """
    SELECT setval(pg_get_serial_sequence('countries', 'country_id'), 6, true);
"""
)

conn.commit()
print("✅ Countries 데이터 삽입 성공!")

cur.execute("SELECT * FROM countries;")
for row in cur.fetchall():
    print(f"  {row}")

cur.close()
conn.close()
