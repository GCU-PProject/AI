# src/core/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, BigInteger, func
from pgvector.sqlalchemy import Vector
from src.core.database import Base


class Law(Base):
    __tablename__ = "laws"

    law_id = Column(BigInteger, primary_key=True, autoincrement=True)
    country_id = Column(BigInteger)
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
