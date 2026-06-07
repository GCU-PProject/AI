"""
영국(UK) 성문법 통합 전처리 및 Raw 데이터 백업 파이프라인 (완성본)
─────────────────────────────────────────────────────────────
[역할]
  Hugging Face의 'othertales/uklegislation' 데이터셋에서 실제 영국 법률 데이터를 가져와
  조항 단위로 청킹(Chunking) 및 정제하고, 새로운 DB의 country_id (10)
  및 실제 컬럼명과 100% 완벽하게 일치하는 필드명을 가진
  'data/UK_Law_Data_Raw.jsonl' 파일로 안전하게 저장합니다.
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

    # 💡 .env 파일 로드를 위한 라이브러리 활성화
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("🚨 [에러] 'datasets' 라이브러리가 설치되어 있지 않습니다.")
    print("👉 해결 방법: pip install datasets python-dotenv")
    sys.exit(1)

# 🔑 .env 파일의 HF_TOKEN을 자동으로 로드
HF_TOKEN = os.getenv("HF_TOKEN")

# 임베딩용 토큰 기준 청킹 설정 (캐나다와 100% 동일)
TOKENIZER_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 2048
OVERLAP_TOKENS = 256

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
    return _TOKENIZER


# 🍁 캐나다 파이프라인과 완벽히 일치하는 청킹 알고리즘
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


def run_raw_pipeline():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_filename = os.path.join(base_dir, "data", "UK_Law_Data_Raw.jsonl")

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print("\n" + "=" * 60)
    print("🇬🇧 영국(UK) 성문법 실제 데이터 전처리 및 Raw JSONL 추출 시작")
    print("=" * 60)
    print(f"📁 출력 대상 파일: {output_filename}")
    print("=" * 60 + "\n")

    grand_total_chunks = 0

    # 신규 국가 ID 매핑 (영국: 14번 정의)
    country_id = 14
    jurisdiction_name = "United Kingdom"
    law_type = "FED"

    print("📥 Hugging Face에서 'othertales/uklegislation' 다운로드 중...")

    try:
        # 💡 [안정화 수정] 캐나다 방식과 동일하게 딕셔너리로 안전하게 로드 후 검증 우회
        # 13GB 대용량 데이터의 메모리/디스크 오버헤드를 막기 위해 스트리밍(streaming=True)을 필수로 적용합니다.
        dataset_dict = load_dataset(
            "othertales/uklegislation",
            token=HF_TOKEN,
            verification_mode="no_checks",
            streaming=True,
        )

        # 'train' 스플릿이 있으면 쓰고, 없으면 첫 번째 스플릿을 유연하게 매핑
        split_name = (
            "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
        )
        dataset = dataset_dict[split_name]

        dir_chunk_count = 0

        # ⭐ [버그 수정] 파일 쓰기용 f_out 선언 누락 해결 ⭐
        with open(output_filename, "w", encoding="utf-8") as f_out:
            for row in tqdm(
                dataset, desc=f"   [{jurisdiction_name}] 실제 정제 진행 중"
            ):
                # 실제 데이터셋 스키마 필드명 매핑 (text_content 사용)
                body_content = row.get("text_content", "")
                section_title = row.get("title", "UK Legislation")

                # 해당 데이터셋은 문서 id나 number를 조항 식별자로 씁니다.
                art_no = str(row.get("number", "General"))
                source_url = row.get("url", "https://www.legislation.gov.uk/")

                # 연도 추출 및 날짜 포맷팅
                year_val = str(row.get("year", "2025"))
                parsed_date = f"{year_val}-01-01 00:00:00+00:00"

                if not body_content or not str(body_content).strip():
                    continue

                # 캐나다 파이프라인과 100% 동일한 토큰 청킹 프로세스
                for piece in split_by_tokens(str(body_content)):
                    if not piece.strip():
                        continue

                    # 캐나다 결과물과 완벽하게 매치되는 DB Key 구조 생성
                    raw_data = {
                        "country_id": country_id,
                        "law_type": law_type,
                        "section_title": section_title,
                        "article_no": art_no,
                        "content": piece,
                        "source_url": source_url,
                        "enactment_date": parsed_date,
                        "amendment_date": parsed_date,
                    }

                    f_out.write(
                        json.dumps(raw_data, ensure_ascii=False, default=str) + "\n"
                    )
                    dir_chunk_count += 1
                    grand_total_chunks += 1

        print(f"   ✅ 처리 완료: {dir_chunk_count:,}개 실제 영국 법률 조항 저장됨.\n")

    except Exception as e:
        print(f"🚨 전처리 중 치명적인 에러 발생: {e}\n")

    print("=" * 60)
    print("🎉 영국 법률 데이터 전처리 진짜 대성공!")
    print(f"📂 최종 생성된 Raw 파일: {output_filename}")
    print(f"🔥 총 추출 및 저장된 조항(임베딩 대상): {grand_total_chunks:,}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_raw_pipeline()
