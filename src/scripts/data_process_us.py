"""
미국 성문법 통합 전처리 및 Raw 데이터 백업 파이프라인 (1단계)
─────────────────────────────────────────────────────────────
[역할]
    미국 주법(CA/NY)을 수집하여 조항 단위로 청킹(Chunking) 및 정제하고,
    DB 컬럼명과 100% 완벽하게 일치하는 필드명을 가진
    'data/US_Law_Data_Raw.jsonl' 파일로 안전하게 저장합니다.
"""

import json
import os
import re
import sys
from tqdm import tqdm
from transformers import AutoTokenizer

# 1. 모듈 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from datasets import load_dataset
except ImportError:
    print("🚨 [에러] 'datasets' 라이브러리가 설치되어 있지 않습니다.")
    print("👉 해결 방법: pip install datasets")
    sys.exit(1)

# 임베딩용 토큰 기준 청킹 설정
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



def iter_statecode_rows(state_code, config_name):
    try:
        dataset_dict = load_dataset(
            "reglab/statecodes",
            config_name,
        )
        dataset = dataset_dict["train"]
        for row in dataset:
            if str(row.get("state", "")).upper() == state_code:
                yield row
        return
    except Exception:
        pass

    try:
        dataset = load_dataset(
            "reglab/statecodes",
            config_name,
            split="train",
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(f"statecodes 데이터셋 로드 실패: {exc}")

    for row in dataset:
        if str(row.get("state", "")).upper() == state_code:
            yield row


def pick_first_text(obj, keys):
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_date_field(row, keys):
    for key in keys:
        val = row.get(key)
        if val:
            return val
    return None


def parse_statecode_section_title(title_text):
    if not title_text:
        return "General"
    parts = [p.strip() for p in title_text.split("›") if p.strip()]
    if not parts:
        return title_text
    for idx, part in enumerate(parts):
        if part.lower().startswith("section "):
            return parts[idx - 1] if idx > 0 else part
    return parts[-1]


def parse_statecode_article_no(title_text):
    if not title_text:
        return "Full Text"
    match = re.search(r"(Section\s+[\w\-\./\(\)]+)", title_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(Rule\s+[\w\-\./\(\)]+)", title_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Full Text"


def extract_state_sections(row):
    section_title = pick_first_text(
        row,
        [
            "section_title",
            "title",
            "chapter",
            "heading",
            "document_title",
        ],
    )

    source_url = pick_first_text(row, ["source_url", "url", "source", "link"])
    enactment_date = extract_date_field(
        row, ["enactment_date", "effective_date", "date"]
    )
    amendment_date = extract_date_field(row, ["amendment_date", "updated_at"])

    sections = row.get("sections") or row.get("section")
    results = []

    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_title = pick_first_text(
                sec, ["section_title", "title", "heading"]
            ) or section_title
            article_no = pick_first_text(
                sec, ["article_no", "section", "section_number", "number"]
            )
            content = pick_first_text(sec, ["content", "text", "body"])
            if content:
                results.append(
                    {
                        "section_title": sec_title or "General",
                        "article_no": article_no or "Full Text",
                        "content": content,
                        "source_url": source_url,
                        "enactment_date": enactment_date,
                        "amendment_date": amendment_date,
                    }
                )
        return results

    if isinstance(sections, dict):
        for key, content in sections.items():
            if not content:
                continue
            results.append(
                {
                    "section_title": section_title or "General",
                    "article_no": str(key),
                    "content": str(content),
                    "source_url": source_url,
                    "enactment_date": enactment_date,
                    "amendment_date": amendment_date,
                }
            )
        return results

    content = pick_first_text(row, ["content", "text", "body"])
    if content:
        title_text = pick_first_text(row, ["title", "path", "section_title"])
        results.append(
            {
                "section_title": parse_statecode_section_title(title_text),
                "article_no": parse_statecode_article_no(title_text),
                "content": content,
                "source_url": source_url,
                "enactment_date": enactment_date,
                "amendment_date": amendment_date,
            }
        )

    return results


def run_raw_pipeline():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_filename = os.path.join(base_dir, "data", "US_Law_Data_Raw.jsonl")

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print("\n" + "=" * 60)
    print("🇺🇸 미국 성문법 통합 전처리 및 Raw JSONL 추출 시작")
    print("=" * 60)
    print(f"📁 출력 대상 파일: {output_filename}")
    print("=" * 60 + "\n")

    grand_total_chunks = 0

    with open(output_filename, "w", encoding="utf-8") as f_out:
        # 1) 미국 주법: California, New York (statecodes)
        for state_name, state_code, country_id, law_type in [
            ("California", "CA", 2, "CA"),
            ("New York", "NY", 3, "NY"),
        ]:
            print(f"📥 Hugging Face에서 '{state_name}' 다운로드 및 스캔 중...")
            try:
                dir_chunk_count = 0
                jurisdiction_name = state_name

                for config_name in ["all_codes", "all_regs"]:
                    rows = iter_statecode_rows(state_code, config_name)

                    for row in tqdm(
                        rows,
                        desc=f"   [{jurisdiction_name}] 정제 진행 중",
                    ):
                        sections = extract_state_sections(row)
                        for sec in sections:
                            content = sec.get("content", "")
                            article_no = sec.get("article_no", "Full Text")
                            section_title = sec.get("section_title", "General")
                            if not content or not str(content).strip():
                                continue

                            for piece in split_by_tokens(str(content)):
                                if not piece.strip():
                                    continue
                                raw_data = {
                                    "country_id": country_id,
                                    "law_type": law_type,
                                    "section_title": section_title,
                                    "article_no": article_no,
                                    "content": piece,
                                    "source_url": sec.get("source_url"),
                                    "enactment_date": sec.get("enactment_date"),
                                    "amendment_date": sec.get("amendment_date")
                                }

                                f_out.write(json.dumps(raw_data, ensure_ascii=False, default=str) + "\n")
                                dir_chunk_count += 1
                                grand_total_chunks += 1

                print(
                    f"   ✅ '{jurisdiction_name}' 처리 완료: {dir_chunk_count:,}개 조항 저장됨.\n"
                )
            except Exception as exc:
                print(f"🚨 '{state_name}' 전처리 중 치명적인 에러 발생: {exc}\n")

    print("=" * 60)
    print("🎉 전처리 및 파일 저장 대성공!")
    print(f"📂 최종 생성된 Raw 파일: {output_filename}")
    print(f"🔥 총 추출 및 저장된 조항(임베딩 대상): {grand_total_chunks:,}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_raw_pipeline()
