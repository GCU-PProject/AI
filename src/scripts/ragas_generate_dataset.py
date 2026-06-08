# src/scripts/ragas_generate_dataset.py
"""
[RAGAS 데이터셋 생성 스크립트]

이 스크립트는 DB(PostgreSQL)에 저장된 법률 문서(chunk)들을 불러와,
LLM을 이용해 RAGAS 평가용 합성 데이터셋(Synthetic Dataset)을 자동 생성합니다.

- LLM: Vertex AI Gemini (gemini-3.5-flash 고정) / Embeddings: 자체 호스팅 Qwen3-Embedding-0.6B
- 소스 데이터: `laws` 테이블의 실제 법률 조항
- 목표 데이터셋 크기: 20개 (주당 5개 × 4개 주, 단일 법령 기반)
  · 일반 사용자(유학생/여행자) 대상이므로, 조항번호를 직접 지목하는 전문가형 질문을
    만들지 않고 seed 법령 하나로 답할 수 있는 질문만 생성
  · seed 법령 1개당 질문 1개 생성 후 주 country_id 태깅
    → 평가 시 실제 서비스처럼 [주 + 연방] 검색 가능

[생성 방식]
- RAGAS TestsetGenerator는 단일 seed 문서에서 sample을 비워 반환하는 경우가 있어,
  평가셋 생성은 직접 LLM 호출로 수행한다.
- 출력 CSV는 RAGAS evaluate가 요구하는 컬럼(user_input, reference_contexts, reference, country_id)을 맞춘다.
"""

import asyncio
import json
import os
import re
import sys

import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import select

from src.core.config import settings
from src.core.database import AsyncSessionLocal
from src.models import Law

# =========================================================
# 1. 설정값 (하이퍼파라미터)
# =========================================================
QUESTIONS_PER_STATE = 5  # 주별 생성 질문 수 → 4개 주면 총 20문제
MAX_GENERATION_RETRIES = 2
OUTPUT_CSV_PATH = "data/ragas_testset.csv"
RAGAS_LLM_MODEL = "gemini-3.5-flash"

# 평가 대상은 '주(state)' 단위로 고정한다. → 고정 벤치마크 + 실제 서비스 정합
# - 실제 서비스는 사용자가 주(예: 캘리포니아)를 선택하면 [주 + 연방]을 함께 검색한다.
#   (chat_service.resolve_jurisdiction_ids) 연방 단독은 사용자가 선택할 일이 없다.
# - 따라서 평가셋도 주(2,3,5,6)만 만들고, 각 주의 질문 생성 재료에 연방법을 함께 포함한다.
#   생성된 질문은 '주 country_id'로 태깅 → 평가 시 resolve_jurisdiction_ids가 [주+연방]으로 검색.
#
#   2=캘리포니아(+1 미국연방), 3=뉴욕(+1), 5=온타리오(+4 캐나다연방), 6=BC(+4)
TARGET_STATE_IDS = [2, 3, 5, 6]

# 각 주에 함께 검색되는 연방 country_id (resolve_jurisdiction_ids와 동일한 매핑)
STATE_TO_FEDERAL = {
    2: 1,  # 캘리포니아 → 미국 연방
    3: 1,  # 뉴욕 → 미국 연방
    5: 4,  # 온타리오 → 캐나다 연방
    6: 4,  # BC → 캐나다 연방
}

# 발표용/고정 평가셋 품질을 높이기 위해 사람이 확인한 좋은 문서를 seed로 먼저 넣는다.
# - 주제: 음주운전 + 세금/체납 + 노동/임금.
# - 생성기는 아래 seed 문서만 사용한다. 랜덤 보충 문서는 쓰지 않는다.
SEED_LAW_IDS_BY_STATE = {
    2: [
        806516,  # CA: 21세 미만 BAC 0.01% 이상 운전 금지
        798031,  # CA: 세금 주간 신고/납부 불이행 시 즉시 납부·징수
        695385,  # CA: 급여 공제액을 직원 단체에 송금해야 하는 기한
        821691,  # CA: 공공계약 일자리 공고와 지원자 우선 추천
        821788,  # CA: 보조금 산정 시 아동부양비 처리
    ],
    3: [
        946257,  # NY: BAC 0.08% / 0.18% 음주운전 기준
        993506,  # NY: 인디언 보호구역 담배세 면제/과세 기준
        924183,  # NY: 고용주의 임금·근로시간 기록 보존 의무
        979990,  # NY: 근무일 spread of hours 정의
        907120,  # NY: 업무상 부상·질병 후 복직/휴직 보호
    ],
    5: [
        1503278,  # Ontario: 초보 운전자 BAC 0 조건
        1361100,  # Canada FED: 세금 체납/출국 우려 시 납부 요구와 압류
        1378631,  # Canada FED: 휴가 중 휴직·질병 사유 발생 시 휴가 중단
        1542494,  # Ontario: 소송 지연 시 사건 기각 기준
        1388237,  # Canada FED: 공공부문 노동 조건 판단 요소
    ],
    6: [
        1574088,  # BC: 운전금지 통지 요건
        1574089,  # BC: 운전금지 통지 후 90일 운전금지
        1361100,  # Canada FED: 세금 체납/출국 우려 시 납부 요구와 압류
        1560922,  # BC: 법정공휴일 근무 시 임금 지급 기준
        1378250,  # Canada FED: 해고된 근로자의 노동 조정 급여 신청
    ],
}

# llm_context: 생성 시 LLM에 주입되는 추가 지시 텍스트 (구버전 language="korean" 대체)
# - 0.4.x에서는 language 파라미터가 제거됨
# - 자유 텍스트 슬롯이므로 ① 서비스/사용자 시나리오 ② 한국어 출력 ③ JSON 형식을 함께 지시
# - 직접 LLM 호출 시 질문/정답 생성 방향을 유도함
# - 단, 질문의 근거는 결국 DB 법률 조항(context)이므로, 조항에 없는 내용을
#   만들지 않도록 강하게 제한한다.
LLM_CONTEXT = (
    # 1) 평가 대상 서비스 & 사용자 시나리오
    "You are generating an evaluation dataset for GLAW, a legal AI assistant "
    "that helps Korean-speaking travelers and residents abroad understand "
    "foreign laws. Its users typically ask practical, real-life questions "
    "about topics such as traffic and DUI rules, visas and immigration, "
    "penalties and fines, and everyday legal issues they may face overseas. "
    "Generate realistic questions that such users would actually ask, "
    "grounded strictly in the provided legal context. "
    "Preferred question style examples: "
    "'캘리포니아에서 렌터카 사고가 났는데 현장에서 바로 합의해도 되나요?', "
    "'뉴욕에서 술을 마시고 운전하면 어느 정도부터 문제가 되나요?', "
    "'온타리오에서 초보 운전자는 술을 조금 마시고 운전해도 되나요?', "
    "'BC에서 운전금지 통지를 받으면 얼마나 운전할 수 없나요?', "
    "'해외에서 세금 체납이 있으면 어떤 절차가 진행되나요?'. "
    "Avoid questions that only ask about register numbers, SOR numbers, repealed sections, "
    "internal government reporting procedures, funding rates, committee procedures, "
    "or administrative approval workflows that ordinary users would rarely ask about. "
    # 2) 언어 규칙 (반드시 한국어 출력)
    "CRITICAL MANDATORY RULE: YOU MUST GENERATE ALL QUESTIONS AND ANSWERS EXCLUSIVELY IN THE KOREAN LANGUAGE (한국어). "
    "Even if instructed to adopt an English-speaking persona or a specific English style (e.g. 'POOR_GRAMMAR'), YOU MUST TRANSLATE your final thought into highly natural Korean before outputting. NEVER use English for the generated questions or answers. "
    # 3) 출력 형식 규칙 (JSON)
    'IMPORTANT: 1) ALWAYS output valid JSON. 2) Ensure internal quotes inside JSON strings are properly escaped (e.g. \\" instead of "). '
    "3) Do not include markdown code block backticks around your JSON payload."
)

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            LLM_CONTEXT
            + " Generate exactly one evaluation sample from the provided legal_text. "
            "You may use ONLY the provided legal_text. Do not use outside legal knowledge, "
            "common sense, assumptions, or facts from other laws. "
            "If the legal_text does not contain enough information for a broad practical question, "
            "create a narrower question that can be answered directly from the legal_text. "
            "The user_input must sound like a practical question from an ordinary Korean user. "
            "Do not mention article numbers, register numbers, or law IDs unless the user would naturally ask them. "
            "The reference must be fully supported by the legal_text and include important numbers, "
            "conditions, periods, or exceptions only when they are present in the legal_text. "
            "Do not add legal advice, penalties, procedures, exceptions, or warnings unless the legal_text explicitly supports them. "
            "The supporting_quote must be a short exact excerpt copied from legal_text that proves the reference. "
            'Return ONLY valid JSON with this exact shape: {{"user_input": "...", "reference": "...", "supporting_quote": "..."}}',
        ),
        (
            "human",
            "law_id: {law_id}\n"
            "jurisdiction_country_id: {country_id}\n"
            "law_type: {law_type}\n"
            "article_no: {article_no}\n"
            "legal_text:\n{legal_text}",
        ),
    ]
)


# =========================================================
# 2. 메인 생성 로직
# =========================================================
async def _fetch_seed_laws(state_id: int, jurisdiction_ids: list[int]):
    seed_ids = SEED_LAW_IDS_BY_STATE.get(state_id, [])
    if not seed_ids:
        return []

    async with AsyncSessionLocal() as session:
        stmt = (
            select(Law)
            .where(Law.country_id.in_(jurisdiction_ids))
            .where(Law.law_id.in_(seed_ids))
        )
        result = await session.execute(stmt)
        laws = result.scalars().all()

    found_ids = {law.law_id for law in laws}
    missing_ids = [law_id for law_id in seed_ids if law_id not in found_ids]
    if missing_ids:
        print(f"      ⚠️ seed 문서 누락: {missing_ids}")

    return laws


async def _fetch_docs_for_state(state_id: int) -> list[Document]:
    """주(state)별 seed 문서만 가져온다.

    seed 문서는 사람이 확인한 문서이며, 실제 서비스 검색 범위([주 + 연방]) 안에 있어야 한다.
    예) 캘리포니아(2) seed는 country_id IN (2, 1) 안에서만 조회.
    """
    jurisdiction_ids = [state_id]
    federal_id = STATE_TO_FEDERAL.get(state_id)
    if federal_id is not None:
        jurisdiction_ids.append(federal_id)

    seed_laws = await _fetch_seed_laws(state_id, jurisdiction_ids)
    seed_ids = {law.law_id for law in seed_laws}
    if seed_laws:
        print(f"      - seed 문서 {len(seed_laws)}개 포함: {sorted(seed_ids)}")

    laws = seed_laws

    docs = []
    for law in laws:
        docs.append(
            Document(
                page_content=law.content,
                metadata={
                    "filename": f"{law.law_type}_{law.article_no}",
                    "law_id": law.law_id,
                    "country_id": law.country_id,
                    "law_type": law.law_type,
                    "article_no": law.article_no,
                },
            )
        )
    return docs


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"JSON 객체를 찾지 못했습니다: {text[:200]}")

    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON 응답이 객체가 아닙니다.")
    required_keys = ["user_input", "reference", "supporting_quote"]
    empty_keys = [key for key in required_keys if not data.get(key)]
    if empty_keys:
        raise ValueError(f"필수 키가 비어 있습니다: {empty_keys} / {data}")
    return data


async def _generate_sample_for_doc(llm, doc: Document, state_id: int) -> dict | None:
    law_id = doc.metadata["law_id"]
    legal_text = _normalize_text(doc.page_content)
    chain = GENERATION_PROMPT | llm | StrOutputParser()

    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            raw = await chain.ainvoke(
                {
                    "law_id": law_id,
                    "country_id": state_id,
                    "law_type": doc.metadata["law_type"],
                    "article_no": doc.metadata["article_no"],
                    "legal_text": legal_text,
                }
            )
            parsed = _parse_json_response(raw)
            supporting_quote = _normalize_text(parsed["supporting_quote"])
            if supporting_quote not in legal_text:
                raise ValueError("supporting_quote가 legal_text 안에 없습니다.")

            return {
                "user_input": _normalize_text(parsed["user_input"]),
                "reference_contexts": json.dumps([legal_text], ensure_ascii=False),
                "reference": _normalize_text(parsed["reference"]),
                "country_id": state_id,
            }
        except Exception as e:
            print(f"      ⚠️ law_id={law_id} 생성 실패 ({attempt}차): {e}")

    return None


async def generate_dataset():
    print("🚀 [1/4] 평가 대상 주(state) 목록을 확인합니다...")
    async with AsyncSessionLocal() as session:
        country_rows = await session.execute(select(Law.country_id).distinct())
        existing_ids = {row[0] for row in country_rows}

    # DB에 실제로 데이터가 있는 주만 대상으로 한다.
    state_ids = [s for s in TARGET_STATE_IDS if s in existing_ids]
    if not state_ids:
        print(f"❌ 대상 주({TARGET_STATE_IDS}) 데이터가 DB에 없습니다.")
        return
    print(f"   📚 평가 대상 주: {state_ids} (각 주 + 연방법 함께 사용)")

    # ── LLM 초기화 (국가 반복과 무관하게 1회만) ──
    print("🧠 [2/4] LLM을 초기화합니다...")
    llm = ChatGoogleGenerativeAI(
        model=RAGAS_LLM_MODEL,  # 발표 전 평가 기준 고정을 위해 명시
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        vertexai=True,
        temperature=0.0,
    )

    # ── seed 법령 1개당 질문 1개 생성 → 주 country_id 태깅 ──
    print(
        f"⚙️ [3/4] 주별 테스트셋 생성 시작! "
        f"(주당 {QUESTIONS_PER_STATE}개 → 총 {len(state_ids) * QUESTIONS_PER_STATE}개 목표)"
    )
    print("   - 생성 방식: seed 법령 1개당 질문 1개")

    all_rows = []
    for state_id in state_ids:
        print(f"\n   🌍 state country_id={state_id} 처리 중...")
        docs = await _fetch_docs_for_state(state_id)
        print(f"      - 재료 문서 {len(docs)}개 추출")
        if not docs:
            print(f"      ⚠️ seed 문서가 없어 건너뜁니다 (country_id={state_id}).")
            continue

        target_docs = docs[:QUESTIONS_PER_STATE]
        if len(target_docs) < QUESTIONS_PER_STATE:
            print(
                f"      ⚠️ 목표 {QUESTIONS_PER_STATE}개보다 seed 문서가 적습니다: "
                f"{len(target_docs)}개"
            )

        state_count = 0
        for doc in target_docs:
            law_id = doc.metadata["law_id"]
            row = await _generate_sample_for_doc(llm, doc, state_id)
            if row is None:
                print(f"      ❌ law_id={law_id} 최종 생성 실패")
                continue

            all_rows.append(row)
            state_count += 1
            print(f"      ✅ law_id={law_id} 질문 1개 생성")

        print(f"      ✅ 총 {state_count}개 질문 생성 (country_id={state_id})")

    if not all_rows:
        print("\n❌ 생성된 질문이 없습니다.")
        return

    # ── 합쳐서 CSV 저장 ──
    print(f"\n💾 [4/4] 전체 결과를 CSV에 저장합니다: {OUTPUT_CSV_PATH}")
    final_df = pd.DataFrame(all_rows)
    keep_columns = ["user_input", "reference_contexts", "reference", "country_id"]
    missing_columns = [col for col in keep_columns if col not in final_df.columns]
    if missing_columns:
        raise ValueError(f"RAGAS 생성 결과에 필요한 컬럼이 없습니다: {missing_columns}")

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    final_df[keep_columns].to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"   🎉 총 {len(final_df)}개 질문 저장 완료 (주 {len(state_ids)}개)")
    print("\n🎉 데이터셋 생성 작업이 성공적으로 종료되었습니다.")
    print("미리보기 (상위 3개):")
    print(final_df[keep_columns].head(3))


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(generate_dataset())
