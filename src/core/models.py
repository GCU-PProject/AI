# src/core/models.py
"""
ORM (Object-Relational Mapping) 모델 정의

이 파일은 데이터베이스 테이블의 구조를 Python 클래스로 정의합니다.
SQLAlchemy ORM을 사용하면 SQL 쿼리를 직접 작성하지 않고,
Python 객체를 다루듯이 DB 데이터를 조회/수정할 수 있습니다.

[ORM이란?]
DB 테이블 ↔ Python 클래스 (1:1 대응)
DB 행(row) ↔ Python 객체 (인스턴스)
DB 컬럼    ↔ 클래스 속성

예시:
    Law 클래스 → laws 테이블
    law = Law(law_type="VEH", article_no="23152.")  → INSERT INTO laws ...
    law.content  → SELECT content FROM laws WHERE ...

[테이블 구조 개요]
1. TestCountry / TestLaw: 초기 개발용 테스트 테이블 (현재 사용하지 않음)
2. Country: 국가/지역 정보 (주 단위 법률 지원)
3. Law: 법률 데이터 + 임베딩 벡터 (RAG 검색의 핵심 테이블)

[관계 구조]
Country (1) ──── (N) Law
하나의 국가/지역은 여러 법률을 가짐
예: California → VEH 23152, PEN 187, CIV 1750, ...
"""

from pgvector.sqlalchemy import Vector
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

# =============================================
# 운영 모델 (현재 사용 중)
# =============================================


class Country(Base):
    """
    국가/지역 정보 테이블

    미국처럼 주(state)마다 법률이 다른 나라를 지원하기 위해
    국가 정보에 주/지역 코드를 추가했습니다.

    [데이터 예시]
    | country_id | country_code | country_name  | state_code | state_name |
    |------------|-------------|---------------|------------|------------|
    | 1          | US          | United States | CA         | California |
    | 2          | US          | United States | NY         | New York   |
    | 3          | GB          | United Kingdom| NULL       | NULL       |
    """

    __tablename__ = "countries"
    x = 1
    # PK: 국가/지역 고유 ID (API 요청 시 country_id로 사용)
    country_id = Column(BigInteger, primary_key=True, autoincrement=True)

    # 국가 코드 (ISO 3166-1 alpha-2, 예: US, GB, JP)
    country_code = Column(String(10), nullable=False)

    # 국가명 (예: United States, United Kingdom)
    country_name = Column(String(100), nullable=False)

    # 주/지역 코드 (예: CA, NY) - 주 단위 법률이 없는 나라는 NULL
    state_code = Column(String(10), nullable=True)

    # 주/지역명 (예: California, New York) - 주 단위 법률이 없는 나라는 NULL
    state_name = Column(String(100), nullable=True)

    # 관리용 날짜 (DB에서 자동 입력)
    # - server_default=func.now(): INSERT 시 DB 서버의 현재 시각 자동 입력
    # - onupdate=func.now(): UPDATE 시 자동 갱신
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 관계 설정 (1:N) - 하나의 국가/지역은 여러 법률을 가짐
    # - relationship(): ORM에서 관계를 정의하는 함수
    # - back_populates="country": Law 모델의 country 속성과 양방향 연결
    # - 사용 예: country.laws → 이 국가의 모든 법률 리스트
    laws = relationship("Law", back_populates="country")

    # 관계 설정 (1:N) - 하나의 국가/지역은 여러 리스크 조합을 가짐
    risks = relationship("Risk", back_populates="country")


class Law(Base):
    """
    법률 데이터 테이블 (RAG 검색의 핵심)

    법률 원문 텍스트와 그 텍스트의 임베딩 벡터를 함께 저장합니다.
    사용자 질문이 들어오면, 질문의 벡터와 이 테이블의 벡터 간
    L2 거리를 계산하여 가장 유사한 법률 조항을 검색합니다.

    [데이터 예시]
    | law_id | country_id | law_type | article_no | content               |
    |--------|-----------|----------|------------|------------------------|
    | 68722  | 1         | VEH      | 23152.     | It is unlawful for...  |
    | 68724  | 1         | VEH      | 23153.     | Any person who...      |
    """

    __tablename__ = "laws"

    # PK: 법률 고유 ID (API 응답의 related_law_id_list에 사용)
    law_id = Column(BigInteger, primary_key=True, autoincrement=True)

    # FK: 이 법률이 속한 국가/지역 ID
    # ForeignKey: 다른 테이블의 PK를 참조하는 외래 키
    # → countries 테이블의 country_id를 참조
    country_id = Column(BigInteger, ForeignKey("countries.country_id"), nullable=False)

    # 법률 종류 코드 (예: VEH=차량법, PEN=형법, CIV=민법)
    # 크롤링 시 법률 코드에서 추출 (초기 law_title에서 law_type으로 변경)
    law_type = Column(String(20), nullable=False)

    # 목차/카테고리 (예: "CHAPTER 1. Offenses Involving Alcohol and Drugs")
    # 법률의 상위 분류를 저장 (구조 파악용)
    section_title = Column(String, nullable=True)

    # 조항 번호 (예: "23152.", "SEC. 1.")
    article_no = Column(String, nullable=False)

    # 법률 본문 텍스트 (RAG의 컨텍스트로 사용되는 핵심 데이터)
    content = Column(Text, nullable=False)

    # 출처 URL (사용자에게 원문 링크를 제공하기 위해 저장)
    source_url = Column(String, nullable=True)

    # 날짜 필드 (크롤링 데이터에 포함되지 않는 경우가 많아 NULL 허용)
    enactment_date = Column(DateTime, nullable=True)  # 제정일
    amendment_date = Column(DateTime, nullable=True)  # 개정일

    # 관리용 날짜 (자동 입력)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 임베딩 벡터 (Vertex AI text-embedding-005 모델로 생성)
    # - Vector(768): pgvector 확장 모듈의 768차원 벡터 타입
    # - 법률 본문(content)을 768개의 숫자 배열로 변환하여 저장
    # - 검색 시 질문 벡터와의 L2 거리(유클리드 거리)를 계산하여 유사도 판단
    embedding = Column(Vector(768))

    # 관계 설정 (N:1) - 이 법률이 속한 국가/지역
    # - back_populates="laws": Country 모델의 laws 속성과 양방향 연결
    # - 사용 예: law.country → 이 법률이 속한 국가 객체
    country = relationship("Country", back_populates="laws")


class Risk(Base):
    """리스크 조합 단위 저장 테이블"""

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
    """리스크 카드 단위 저장 테이블"""

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
