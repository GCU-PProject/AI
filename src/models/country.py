# src/models/country.py

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    String,
    func,
)
from sqlalchemy.orm import relationship

from src.core.database import Base


class Country(Base):
    """
    국가/지역 정보 테이블

    [데이터 예시]
    | country_id | country_code | country_name  | state_code | state_name |
    |------------|-------------|---------------|------------|------------|
    | 1          | US          | United States | CA         | California |
    | 2          | US          | United States | NY         | New York   |
    | 3          | GB          | United Kingdom| NULL       | NULL       |
    """

    __tablename__ = "countries"

    # PK: 국가/지역 고유 ID
    country_id = Column(BigInteger, primary_key=True, autoincrement=True)
    # 국가 코드
    country_code = Column(String(10), nullable=False)
    # 국가명
    country_name = Column(String(100), nullable=False)
    # 주/지역 코드 (
    state_code = Column(String(10), nullable=True)
    # 주/지역명
    state_name = Column(String(100), nullable=True)
    # 관리용 날짜 (DB에서 자동 입력)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # 관계 설정 (1:N)
    laws = relationship("Law", back_populates="country")
    risks = relationship("Risk", back_populates="country")
