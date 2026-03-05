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
        (1, 'US', 'CA', 'United States', 'California'),
        (2, 'CA', NULL, 'Canada', NULL)
    ON CONFLICT (country_id) DO NOTHING;
"""
)

conn.commit()
print("✅ Countries 데이터 삽입 성공!")

cur.execute("SELECT * FROM countries;")
for row in cur.fetchall():
    print(f"  {row}")

cur.close()
conn.close()
