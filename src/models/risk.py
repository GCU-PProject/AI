# src/models/risk.py

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from src.core.database import Base


class Risk(Base):
    """
    리스크 조합 단위 저장 테이블

    국가, 여행 목적, 비자 종류, 연령대를 하나의 조건 조합으로 저장합니다.
    각 조합은 여러 개의 리스크 카드(RiskList)를 가질 수 있습니다.

    [데이터 예시]
    | risk_id | country_id | travel_purpose | visa_type  | age_band | overall_risk_level |
    |---------|------------|----------------|------------|----------|--------------------|
    | 1       | 1          | tourism        | short_stay | 20s      | MEDIUM             |
    | 2       | 1          | work           | work_permit| 30s      | HIGH               |
    """

    __tablename__ = "risk"
    # 동일 국가/조건 조합은 1건만 유지
    __table_args__ = (
        UniqueConstraint(
            "country_id",
            "travel_purpose",
            "visa_type",
            "age_band",
            name="uq_risk_country_purpose_visa_age",
        ),
    )

    risk_id = Column(BigInteger, primary_key=True, autoincrement=True)
    # 조회 조건으로 사용하는 사용자 상황 필드
    country_id = Column(BigInteger, ForeignKey("countries.country_id"), nullable=False)
    travel_purpose = Column(String(50), nullable=False)
    visa_type = Column(String(50), nullable=False)
    age_band = Column(String(20), nullable=False)
    # 카드 전체를 대표하는 상위 위험도
    overall_risk_level = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 국가-리스크 조합 / 조합-카드(1:N) 관계
    country = relationship("Country", back_populates="risks")
    risk_items = relationship("RiskList", back_populates="risk")


class RiskList(Base):
    """
    리스크 카드 단위 저장 테이블

    하나의 리스크 조합(Risk)에 속하는 개별 리스크 카드 정보를 저장합니다.
    화면에는 sort_order 순서대로 여러 개의 카드가 노출됩니다.

    [데이터 예시]
    | risk_list_id | risk_id | sort_order | risk_title        | risk_level |
    |--------------|---------|------------|-------------------|------------|
    | 1            | 1       | 1          | 음주 관련 법적 위험 | HIGH       |
    | 2            | 1       | 2          | 교통 법규 위반 위험 | MEDIUM     |
    """

    __tablename__ = "risk_list"
    # 같은 리스크 조합 내에서 정렬 순서 중복 방지
    __table_args__ = (
        UniqueConstraint("risk_id", "sort_order", name="uq_risk_list_risk_id_order"),
    )

    risk_list_id = Column(BigInteger, primary_key=True, autoincrement=True)
    risk_id = Column(BigInteger, ForeignKey("risk.risk_id"), nullable=False)
    # 화면 노출 순서를 고정하기 위한 정렬 값
    sort_order = Column(BigInteger, nullable=False)
    risk_title = Column(String(255), nullable=False)
    risk_level = Column(String(20), nullable=False)
    risk_content = Column(Text, nullable=False)
    # 배열 구조를 그대로 저장하기 위해 JSON 사용
    risk_actions = Column(JSON, nullable=False, default=list)
    law_refs = Column(JSON, nullable=False, default=list)
    issue_refs = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    risk = relationship("Risk", back_populates="risk_items")
