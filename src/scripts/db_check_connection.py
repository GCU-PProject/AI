import sys
import os

# dirname을 3번 중첩해서 루트까지
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src.core.config import settings

DATABASE_URL = settings.DATABASE_URL

print(f"--- 접속 정보 확인 ---")
print(f"HOST: {settings.DB_HOST}")
print(f"USER: {settings.DB_USER}")
print(f"DB  : {settings.DB_NAME}")
print("--------------------")


def test_connection():
    try:
        # 엔진 생성
        engine = create_engine(DATABASE_URL)

        # 연결 시도
        with engine.connect() as connection:
            # 간단한 쿼리 실행 (1을 반환하는 쿼리)
            result = connection.execute(text("SELECT 1"))
            print("\n✅ [대성공] 데이터베이스 연결에 성공했습니다!")
            print("   터널링이 완벽하게 작동하고 있습니다.")
            print(f"   테스트 쿼리 결과: {result.scalar()}")

    except OperationalError as e:
        print("\n❌ [실패] 데이터베이스 연결에 실패했습니다.")
        print(f"   에러 내용: {e}")
    except Exception as e:
        print("\n❌ [오류] 알 수 없는 오류가 발생했습니다.")
        print(f"   {e}")


if __name__ == "__main__":
    test_connection()
