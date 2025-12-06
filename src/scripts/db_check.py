import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# 1. .env 파일 로드
load_dotenv()

# 2. 환경 변수 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 3. DB URL 생성
# 포맷: postgresql://아이디:비번@주소:포트/DB이름
# 뒤에 ?sslmode=require 추가 (보안 연결 강제 설정)
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"  # disable

print(f"--- 접속 정보 확인 ---")
print(f"HOST: {DB_HOST}")
print(f"USER: {DB_USER}")
print(f"DB  : {DB_NAME}")
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
        print("   [체크리스트]")
        print("   1. 터널링 터미널이 켜져 있나요?")
        print("   2. .env 파일의 DB_PASSWORD가 정확한가요?")
        print(f"   에러 내용: {e}")
    except Exception as e:
        print("\n❌ [오류] 알 수 없는 오류가 발생했습니다.")
        print(f"   {e}")


if __name__ == "__main__":
    test_connection()
