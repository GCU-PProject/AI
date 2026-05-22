"""
캐나다 성문법 통합 전처리 및 Raw 데이터 백업 파이프라인 (1단계)
─────────────────────────────────────────────────────────────
[역할]
  Hugging Face에서 캐나다 법률(6대 카테고리) 데이터를 긁어와서
  조항 단위로 청킹(Chunking) 및 정제하고, 새로운 DB의 country_id (4, 5, 6)
  및 실제 컬럼명과 100% 완벽하게 일치하는 필드명을 가진
  'data/Canada_Law_Data_Raw.jsonl' 파일로 안전하게 저장합니다.
"""

import json
import os
import re
import sys
from tqdm import tqdm

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

# 6대 세부 법률 디렉토리 정의
CA_DATA_DIRS = [
    "LEGISLATION-FED",
    "REGULATIONS-FED",
    "LEGISLATION-ON",
    "REGULATIONS-ON",
    "LEGISLATION-BC",
    "REGULATIONS-BC",
]


def get_section_title(unofficial_text_en: str) -> str:
    """원문 마크다운에서 가장 처음에 등장하는 대표 헤더(## 또는 #)를 카테고리명으로 지정"""
    if not unofficial_text_en:
        return "General"
    match = re.search(r"^(?:#|##)\s+(.+)$", unofficial_text_en, re.MULTILINE)
    return match.group(1).strip() if match else "General"


def run_raw_pipeline():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_filename = os.path.join(base_dir, "data", "Canada_Law_Data_Raw.jsonl")

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print("\n" + "=" * 60)
    print("🍁 캐나다 성문법 통합 전처리 및 Raw JSONL 추출 시작")
    print("=" * 60)
    print(f"📁 출력 대상 파일: {output_filename}")
    print("=" * 60 + "\n")

    grand_total_chunks = 0

    with open(output_filename, "w", encoding="utf-8") as f_out:
        for data_dir in CA_DATA_DIRS:
            print(f"📥 Hugging Face에서 '{data_dir}' 다운로드 및 스캔 중...")
            
            # ⭐ [핵심 변경] 국가/주 테이블 신규 ID 매핑 반영 (4, 5, 6번) ⭐
            dir_upper = data_dir.upper()
            if "FED" in dir_upper:
                country_id = 4             # 캐나다 연방 (Federal)
                jurisdiction_name = "Federal"
                law_type = "FED"
            elif "ON" in dir_upper:
                country_id = 5             # 온타리오 주 (Ontario)
                jurisdiction_name = "Ontario"
                law_type = "ON"
            elif "BC" in dir_upper:
                country_id = 6             # 브리티시 컬럼비아 주 (British Columbia)
                jurisdiction_name = "British Columbia"
                law_type = "BC"
            else:
                continue

            try:
                dataset_dict = load_dataset("a2aj/canadian-laws", data_dir=data_dir)
                dataset = dataset_dict["train"]
                
                dir_chunk_count = 0

                for row in tqdm(dataset, desc=f"   [{jurisdiction_name}] 정제 진행 중"):
                    unofficial_text_en = row.get("unofficial_text_en", "")
                    unofficial_sections_en = row.get("unofficial_sections_en", None)
                    source_url = row.get("source_url_en", "")

                    if isinstance(unofficial_sections_en, str) and unofficial_sections_en.strip():
                        try:
                            unofficial_sections_en = json.loads(unofficial_sections_en)
                        except Exception:
                            unofficial_sections_en = None

                    if not unofficial_sections_en or not isinstance(unofficial_sections_en, dict):
                        continue

                    # 날짜 취득
                    parsed_date = row.get("document_date_en", None)
                    # 첫 번째 대표 헤더로 대분류 제목 추출
                    section_title = get_section_title(unofficial_text_en)

                    for art_no, body_content in unofficial_sections_en.items():
                        if not body_content or not str(body_content).strip():
                            continue

                        # DB 모델 컬럼명과 100% 완벽 통합된 키 이름 설정
                        raw_data = {
                            "country_id": country_id,
                            "law_type": law_type,
                            "section_title": section_title,
                            "article_no": art_no,
                            "content": body_content,
                            "source_url": source_url,
                            "enactment_date": parsed_date,
                            "amendment_date": parsed_date
                        }

                        f_out.write(json.dumps(raw_data, ensure_ascii=False, default=str) + "\n")
                        dir_chunk_count += 1
                        grand_total_chunks += 1

                print(f"   ✅ '{data_dir}' 처리 완료: {dir_chunk_count:,}개 조항 저장됨.\n")

            except Exception as e:
                print(f"🚨 '{data_dir}' 전처리 중 치명적인 에러 발생: {e}\n")

    print("=" * 60)
    print("🎉 전처리 및 파일 저장 대성공!")
    print(f"📂 최종 생성된 Raw 파일: {output_filename}")
    print(f"🔥 총 추출 및 저장된 조항(임베딩 대상): {grand_total_chunks:,}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_raw_pipeline()
