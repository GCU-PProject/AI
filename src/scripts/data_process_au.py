"""
호주(AU) 성문법 통합 전처리 및 데이터 가공 파이프라인
"""

import json
import os
import sys
import re
from transformers import AutoTokenizer

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 1. 모듈 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 임베딩용 토큰 기준 청킹 설정 (표준화)
TOKENIZER_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 2048
OVERLAP_TOKENS = 256

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
    return _TOKENIZER


def split_by_tokens(text, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    if not text:
        return []
    if overlap >= max_tokens:
        overlap = max_tokens // 4

    tokenizer = get_tokenizer()
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return [text]

    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(token_ids), step):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        if not chunk_ids:
            break
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(token_ids):
            break

    return chunks

# 호주 관할권별 촘촘한 고유 country_id 매핑 설계 (7~13번)
JURISDICTION_MAP = {
    "commonwealth": 7,       # 호주 연방
    "new_south_wales": 8,    # 뉴사우스웨일스 (NSW)
    "queensland": 9,         # 퀸즐랜드 (QLD)
    "western_australia": 10, # 서호주 (WA)
    "south_australia": 11,   # 남호주 (SA)
    "tasmania": 12,          # 태즈메이니아 (TAS)
    "norfolk_island": 13,    # 노퍽 섬 (NF)
}

LAW_TYPE_MAP = {
    7: "FED",
    8: "NSW",
    9: "QLD",
    10: "WA",
    11: "SA",
    12: "TAS",
    13: "NF"
}


# =========================================================
# 1. 텍스트 분리 및 정제
# =========================================================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_articles(text):
    chunks = []
    pattern = r"^(\d+[A-Z]*\.?\s+|\d+[A-Z]*\u2014)([A-Z].+)"
    lines = text.split("\n")
    curr_art = None
    curr_content_lines = []

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            if curr_content_lines:
                content = clean_text("\n".join(curr_content_lines))
                if content and len(content) >= 50:
                    if not curr_art:
                        curr_art = "General Provisions"
                    chunks.append({"article_no": curr_art, "content": content})
            art_num = match.group(1).strip().rstrip(".")
            art_title = match.group(2).strip()
            curr_art = f"{art_num}. {art_title}"
            curr_content_lines = []
        else:
            curr_content_lines.append(line)

    if curr_content_lines:
        content = clean_text("\n".join(curr_content_lines))
        if content and len(content) >= 50:
            if not curr_art:
                curr_art = "General Provisions"
            chunks.append({"article_no": curr_art, "content": content})

    if not chunks:
        cleaned_full = clean_text(text)
        if cleaned_full and len(cleaned_full) >= 50:
            chunks.append({"article_no": "Full Text", "content": cleaned_full})

    return chunks


# =========================================================
# 2. 데이터셋 다운로드 및 전처리
# =========================================================
def export_to_jsonl():
    print("=" * 60)
    print("🇦🇺 호주 성문법 데이터셋 다운로드 및 가공 시작")
    print("=" * 60)

    try:
        from datasets import load_dataset
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("🚨 [에러] 'datasets' 및 'python-dotenv' 라이브러리가 필요합니다.")
        return

    # .env 파일의 HF_TOKEN 로드
    hf_token = os.getenv("HF_TOKEN")

    print("\n📊 HuggingFace 데이터셋 로딩 중...")
    try:
        ds = load_dataset(
            "isaacus/open-australian-legal-corpus",
            token=hf_token,
            verification_mode="no_checks"
        )
        corpus = ds["corpus"]
    except Exception as e:
        print(f"🚨 데이터셋 다운로드 실패: {e}")
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_filename = os.path.join(base_dir, "data", "Australia_Law_Data_Raw.jsonl")
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    processed_count = 0
    filtered_count = 0
    skipped_count = 0

    with open(output_filename, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(corpus):
            doc_type = item.get("type", "")
            jurisdiction = item.get("jurisdiction", "unknown")

            # primary_legislation 필터링
            if doc_type != "primary_legislation" or jurisdiction not in JURISDICTION_MAP:
                skipped_count += 1
                continue

            text = item.get("text", "")
            url = item.get("url", "")
            section_title = item.get("citation", "")
            
            country_id = JURISDICTION_MAP.get(jurisdiction, 7)
            law_type = LAW_TYPE_MAP.get(country_id, "FED")

            filtered_count += 1
            sections = split_into_articles(text)
            
            for sec in sections:
                for piece in split_by_tokens(sec["content"]):
                    row = {
                        "country_id": country_id,
                        "law_type": law_type,
                        "section_title": section_title,
                        "article_no": sec["article_no"],
                        "content": piece,
                        "source_url": url,
                        "enactment_date": None,
                        "amendment_date": None
                    }
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed_count += 1

    print(f"\n✅ 완료! 가공된 조항 수: {processed_count:,}개")


if __name__ == "__main__":
    export_to_jsonl()
