# src/core/models.py
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    BigInteger,
    func,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.core.database import Base


class Country(Base):
    __tablename__ = "test_countries"

    # PK: 국가 고유 ID
    country_id = Column(BigInteger, primary_key=True, autoincrement=True)

    # 국가 코드 (예: KR, US, GB, SG)
    country_code = Column(String(10), unique=True, nullable=False)

    # 국가명 (예: 대한민국, United Kingdom)
    country_name = Column(String(100), nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 관계 설정 (1:N) - 하나의 국가는 여러 법률을 가짐
    laws = relationship("Law", back_populates="country")


class Law(Base):
    __tablename__ = "test_laws"

    law_id = Column(BigInteger, primary_key=True, autoincrement=True)

    country_id = Column(
        BigInteger, ForeignKey("test_countries.country_id"), nullable=False
    )

    law_title = Column(String)
    category = Column(String)
    article_no = Column(String)
    content = Column(Text)
    enactment_date = Column(DateTime)
    amendment_date = Column(DateTime)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 768차원 벡터
    embedding = Column(Vector(768))


class Law(Base):
    __tablename__ = "laws"  # 테이블 이름 (기존 test_laws -> laws 변경 권장)

    # 1. 기본 키 (BigInteger 유지)
    law_id = Column(BigInteger, primary_key=True, autoincrement=True)

    # 2. 국가 코드 (미국=1)
    country_id = Column(BigInteger, nullable=False)

    # 3. [변경] 법령 종류 (예: VEH, CIV, PEN)
    # 기존 law_title -> law_type으로 변경하여 코드값 저장
    law_type = Column(String(20), nullable=False)

    # 4. [변경] 목차/카테고리 정보 (예: CHAPTER 1. Reports...)
    # 기존 category 컬럼 유지 (길이 제한 없음)
    section_title = Column(String, nullable=True)

    # 5. 조항 번호 (예: 23152., SEC. 1.)
    article_no = Column(String, nullable=False)

    # 6. 본문 내용
    content = Column(Text, nullable=False)

    # 7. [신규] 출처 URL (중요!)
    # RAG 답변 시 사용자에게 링크를 제공하기 위해 필수입니다.
    source_url = Column(String, nullable=True)

    # 8. [변경] 날짜 필드 (수집 데이터에 없으므로 Null 허용)
    # 필요하다면 남겨두되, nullable=True로 설정합니다.
    enactment_date = Column(DateTime, nullable=True)
    amendment_date = Column(DateTime, nullable=True)

    # 9. 관리용 날짜 (자동 입력)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 10. 임베딩 벡터 (Vertex AI text-embedding-005 기준 768차원)
    embedding = Column(Vector(768))
