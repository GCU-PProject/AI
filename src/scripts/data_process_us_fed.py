"""
미국 연방법(US Code) 통합 전처리 및 Raw 데이터 백업 파이프라인 (1단계)
─────────────────────────────────────────────────────────────
[역할]
  Hugging Face의 emre570/us-legal-code 데이터를 사용하여
  조항 단위로 청킹(Chunking) 및 정제하고,
  DB 컬럼명과 100% 완벽하게 일치하는 필드명을 가진
  'data/US_Fed_Law_Data_Raw.jsonl' 파일로 안전하게 저장합니다.
"""

import json
import os
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


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value


def run_raw_pipeline():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_filename = os.path.join(base_dir, "data", "US_Fed_Law_Data_Raw.jsonl")

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print("\n" + "=" * 60)
    print("🇺🇸 미국 연방법(US Code) 전처리 및 Raw JSONL 추출 시작")
    print("=" * 60)
    print(f"📁 출력 대상 파일: {output_filename}")
    print("=" * 60 + "\n")

    grand_total_chunks = 0

    with open(output_filename, "w", encoding="utf-8") as f_out:
        print("📥 Hugging Face에서 'emre570/us-legal-code' 다운로드 및 스캔 중...")

        try:
            dataset = load_dataset("emre570/us-legal-code", split="train")

            dir_chunk_count = 0
            country_id = 1
            law_type = "FED"

            for row in tqdm(dataset, desc="   [US FED] 정제 진행 중"):
                section_id = normalize_value(row.get("section_id"))
                heading = normalize_value(row.get("heading"))
                text = normalize_value(row.get("text"))
                source_url = normalize_value(row.get("section_url"))

                if not text or not str(text).strip():
                    continue

                section_title = heading or "General"
                article_no = f"Section {section_id}" if section_id else "Full Text"

                for piece in split_by_tokens(str(text)):
                    if not piece.strip():
                        continue

                    raw_data = {
                        "country_id": country_id,
                        "law_type": law_type,
                        "section_title": section_title,
                        "article_no": article_no,
                        "content": piece,
                        "source_url": source_url,
                        "enactment_date": None,
                        "amendment_date": None
                    }

                    f_out.write(json.dumps(raw_data, ensure_ascii=False, default=str) + "\n")
                    dir_chunk_count += 1
                    grand_total_chunks += 1

            print(f"   ✅ 'US FED' 처리 완료: {dir_chunk_count:,}개 조항 저장됨.\n")

        except Exception as e:
            print(f"🚨 'US FED' 전처리 중 치명적인 에러 발생: {e}\n")

    print("=" * 60)
    print("🎉 전처리 및 파일 저장 대성공!")
    print(f"📂 최종 생성된 Raw 파일: {output_filename}")
    print(f"🔥 총 추출 및 저장된 조항(임베딩 대상): {grand_total_chunks:,}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_raw_pipeline()
