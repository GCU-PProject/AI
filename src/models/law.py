# src/models/law.py

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from src.core.database import Base


class Law(Base):
    """
    법률 데이터 테이블

    [데이터 예시]
    | law_id | country_id | law_type | article_no | content               |
    |--------|-----------|----------|------------|------------------------|
    | 68722  | 2         | VEH      | 23152.     | It is unlawful for...  |
    | 68724  | 2         | VEH      | 23153.     | Any person who...      |
    (country_id=2: 캘리포니아 — countries 테이블 참고)
    """

    __tablename__ = "laws"

    # PK: 법률 고유 ID
    law_id = Column(BigInteger, primary_key=True, autoincrement=True)
    # FK: 이 법률이 속한 국가/지역 ID
    country_id = Column(BigInteger, ForeignKey("countries.country_id"), nullable=False)
    # 법률 종류 코드 (예: VEH=차량법, PEN=형법, CIV=민법)
    law_type = Column(String(20), nullable=False)
    # 목차/카테고리 (예: "CHAPTER 1. Offenses Involving Alcohol and Drugs")
    section_title = Column(String, nullable=True)
    # 조항 번호 (예: "23152.", "SEC. 1.")
    article_no = Column(String, nullable=False)
    # 법률 본문 텍스트
    content = Column(Text, nullable=False)
    # 출처 URL
    source_url = Column(String, nullable=True)
    # 날짜 필드 (타임존 보존)
    enactment_date = Column(DateTime(timezone=True), nullable=True)  # 제정일
    amendment_date = Column(DateTime(timezone=True), nullable=True)  # 개정일
    # 관리용 날짜 (DB에서 자동 입력)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # 임베딩 벡터 (Qwen3-Embedding-0.6B 모델로 생성)
    embedding = Column(Vector(1024))
    # 관계 설정 (N:1)
    country = relationship("Country", back_populates="laws")
