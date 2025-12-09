# src/core/models.py
from sqlalchemy import (
    Column,
    Integer,
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

    country = relationship("Country", back_populates="laws")
