"""
호주 법률 데이터 가공 스크립트

HuggingFace의 호주 공개 법률 데이터셋(isaacus/open-australian-legal-corpus)을
다운로드하고, 법률(primary_legislation)과 하위법령(secondary_legislation)만 필터링한 뒤,
조항 단위로 분리하여 data/law_data_AU.jsonl 파일로 출력합니다.

[원본 데이터 필드]
- version_id: 문서 고유 ID
- type: 문서 유형 (primary_legislation, secondary_legislation, bill, decision)
- jurisdiction: 관할권 (commonwealth, new_south_wales, queensland 등)
- source: 데이터 출처 사이트
- citation: 법률 제목/인용명
- url: 원본 출처 URL
- date: 제정/판결 날짜
- text: 법률 본문 전체 텍스트

[처리 흐름]
1. HuggingFace 데이터셋 로드
2. 법률 + 하위법령만 필터링 (bill, decision 제외)
3. 법률 텍스트를 조항(Section) 단위로 분리
4. JSONL 파일로 저장

[실행 방법]
python src/scripts/data_process_au.py
"""

import json
import os
import sys
import re

# 프로젝트 루트를 sys.path에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from datasets import load_dataset

# 처리 대상 문서 유형 (본법만 유지, 하위법령 제외)
VALID_TYPES = {"primary_legislation"}

# 대상 관할권 (MVP 타겟: 캘리포니아와 대응되는 NSW)
TARGET_JURISDICTION = "new_south_wales"


# =========================================================
# 1. 텍스트 처리 함수
# =========================================================
def clean_text(text):
    """공백을 정리하고 앞뒤 여백을 제거합니다."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_articles(text):
    """
    법률 텍스트를 조항(Section) 단위로 분리합니다.

    호주 각 주(jurisdiction)별 조항 헤더 형식:
      - Commonwealth/NSW/QLD: "1 Short title" (번호 + 공백)
      - Norfolk Island/WA:    "1. Short title" (번호 + 점 + 공백)
      - Tasmania:             "1.   Short title" (번호 + 점 + 공백 여러 개)
      - South Australia:      "1—Short title" (번호 + em dash) 또는 탭 구분
      - 알파벳 접미사:         "8A", "24AA", "55N" 등

    줄 시작(^)에 있는 헤더만 매칭하여, 본문 내 참조
    ("under section 51 of the Constitution" 등)에서 잘리는 문제를 방지합니다.
    50자 미만의 짧은 조항은 제외합니다.
    """
    chunks = []

    # 줄 시작 + 번호(+선택 알파벳) + 구분자(점/em dash/공백/탭) + 대문자 제목
    pattern = r"^(\d+[A-Z]*\.?\s+|\d+[A-Z]*\u2014)([A-Z].+)"

    # 텍스트를 줄 단위로 처리
    lines = text.split("\n")
    curr_art = None
    curr_content_lines = []

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            # 이전 조항 저장
            if curr_content_lines:
                content = clean_text("\n".join(curr_content_lines))
                if content and len(content) >= 50:
                    if not curr_art:
                        curr_art = "General Provisions"
                    chunks.append({"article_no": curr_art, "content": content})

            # 새 조항 시작
            art_num = match.group(1).strip().rstrip(".")
            art_title = match.group(2).strip()
            curr_art = f"{art_num}. {art_title}"
            curr_content_lines = []
        else:
            curr_content_lines.append(line)

    # 마지막 조항 저장
    if curr_content_lines:
        content = clean_text("\n".join(curr_content_lines))
        if content and len(content) >= 50:
            if not curr_art:
                curr_art = "General Provisions"
            chunks.append({"article_no": curr_art, "content": content})

    # 패턴으로 하나도 못 쪼갰으면 전체를 하나로
    if not chunks:
        cleaned_full = clean_text(text)
        if cleaned_full and len(cleaned_full) >= 50:
            chunks.append({"article_no": "Full Text", "content": cleaned_full})

    return chunks


# =========================================================
# 2. 메인 실행 함수
# =========================================================
def export_to_jsonl():
    print("=" * 60)
    print("🇦🇺 호주 법률 데이터 가공 시작 (NSW 본법 전용)")
    print("=" * 60)

    # ----- Step 1: HuggingFace 데이터셋 로드 -----
    print("\n📊 [1/3] HuggingFace 데이터셋 로딩 중...")
    ds = load_dataset("isaacus/open-australian-legal-corpus")

    corpus = ds["corpus"]
    print(f"   ✅ 전체 {len(corpus)}개 문서 로드 완료")

    # ----- Step 2: 필터링 + 조항 분리 -----
    print(f"\n📋 [2/3] NSW 본법(Primary Legislation)만 필터링 후 조항 분리 중...")
    print(f"   대상 유형: {VALID_TYPES}, 관할권: {TARGET_JURISDICTION}")

    output_filename = "data/law_data_AU.jsonl"
    processed_count = 0
    filtered_count = 0
    skipped_count = 0

    # 호주 관할권별 country_id 매핑 (DB countries 테이블과 일치)
    jurisdiction_map = {
        "commonwealth": 3,  # 호주 연방
        "new_south_wales": 4,  # 뉴사우스웨일스
        "queensland": 5,  # 퀸즐랜드
        "western_australia": 6,  # 서호주
        "south_australia": 7,  # 남호주
        "tasmania": 8,  # 태즈메이니아
        "norfolk_island": 9,  # 노퍽 섬
    }

    with open(output_filename, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(corpus):
            doc_type = item.get("type", "")
            jurisdiction = item.get("jurisdiction", "unknown")

            # 1. 1차 법률(Primary)이 아니면 패스
            # 2. 지정된 관할권(NSW)이 아니면 패스
            if doc_type not in VALID_TYPES or jurisdiction != TARGET_JURISDICTION:
                skipped_count += 1
                continue

            text = item.get("text", "")
            url = item.get("url", "")
            section_title = item.get("citation", "")
            jurisdiction = item.get("jurisdiction", "unknown")

            # country_id 매핑 (매칭 안 되면 기본값 10)
            country_id = jurisdiction_map.get(jurisdiction, 10)

            filtered_count += 1

            # 진행률 표시
            if filtered_count % 500 == 0:
                print(
                    f"   📌 진행: {filtered_count}건 처리 / {skipped_count}건 건너뜀 ({idx+1}/{len(corpus)})"
                )

            # 조항 단위로 분리
            sections = split_into_articles(text)
            for sec in sections:
                row = {
                    "country_id": country_id,
                    "law_type": doc_type,
                    "section_title": section_title,
                    "article_no": sec["article_no"],
                    "content": sec["content"],
                    "source_url": url,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed_count += 1

    # ----- Step 3: 완료 -----
    print(f"\n{'=' * 60}")
    print(f"✅ 완료! '{output_filename}'")
    print(f"   📄 처리된 문서: {filtered_count}개 (전체 {filtered_count + skipped_count}개 중)")
    print(f"   🚫 건너뛴 문서: {skipped_count}개 (bill, decision)")
    print(f"   📝 생성된 조항: {processed_count}개")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    export_to_jsonl()
