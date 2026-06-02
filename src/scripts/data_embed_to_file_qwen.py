"""
Qwen3-Embedding-0.6B 로컬 임베딩 스크립트 (2단계)
───────────────────────────────────────────────────────────────────
[역할]
  1단계에서 생성된 Raw JSONL 파일(*_Raw.jsonl)을 읽어,
  Qwen3-Embedding-0.6B 모델로 1024차원 벡터를 생성한 뒤
  기존 필드에 'embedding' 필드만 추가하여
  새로운 *_Embedded.jsonl 파일로 저장합니다.

[입력] data/*_Raw.jsonl
[출력] data/*_Embedded.jsonl
"""

import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 1. 모듈 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 2. 실행 디바이스 설정 (MPS/CUDA/CPU)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🖥️  활성화된 하드웨어 가속 디바이스: [{device.upper()}]")

# 3. Qwen3-Embedding-0.6B 모델 적재
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
print(f"🔄 {MODEL_ID} 모델 로딩 중 (VRAM 적재)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
).to(device)
model.eval()
print("✅ 모델 로딩 대성공!")


def _last_token_pool(last_hidden_states, attention_mask):
    """Last-token pooling (Qwen3-Embedding 공식 권장 방식).

    각 문장에서 '마지막 실제 토큰'의 벡터를 문장 표현으로 사용한다.
    좌측 패딩/우측 패딩 모두를 안전하게 처리한다.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lengths]


def get_embeddings_qwen(texts):
    """Qwen3 모델로 '문서(passage)' 배치의 1024차원 임베딩을 추출한다.

    [Qwen3 공식 가이드 준수]
    - 문서(passage)에는 instruction(접두사)을 붙이지 않고 원문 그대로 임베딩한다.
    - pooling은 last-token pooling을 사용한다.
    - 질문(query) 임베딩(embed_server.py)에만 Instruct 형식을 적용한다.
    """
    encoded_input = tokenizer(
        texts,  # 문서는 접두사 없이 원문 그대로
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
        token_embeddings = model_output[0]
        attention_mask = encoded_input["attention_mask"]

        # Last-token pooling
        embeddings = _last_token_pool(token_embeddings, attention_mask)

        # L2 정규화 (Cosine Similarity 호환)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().float().numpy().tolist()


def run_embed_pipeline():
    """Raw JSONL 파일을 읽어 임베딩을 추가한 Embedded JSONL 파일 생성"""
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    raw_files = glob.glob(os.path.join(base_dir, "data", "*_Raw.jsonl"))

    if not raw_files:
        print("🚨 'data/' 디렉토리에 '_Raw.jsonl' 파일이 존재하지 않습니다.")
        print("👉 먼저 1단계(data_process_ca.py)를 실행하여 Raw 파일을 생성하세요.")
        return

    print(f"📂 임베딩 대상 파일 목록: {[os.path.basename(f) for f in raw_files]}")

    for raw_filepath in raw_files:
        # 출력 파일명: *_Raw.jsonl → *_Embedded.jsonl
        embedded_filepath = raw_filepath.replace("_Raw.jsonl", "_Embedded.jsonl")
        print(f"\n{'=' * 60}")
        print(f"🚀 [임베딩 시작] {os.path.basename(raw_filepath)}")
        print(f"📁 출력 파일: {os.path.basename(embedded_filepath)}")
        print(f"{'=' * 60}")

        # 전체 라인 수 카운트 (진행률 표시용)
        with open(raw_filepath, "r", encoding="utf-8") as f_cnt:
            total_lines = sum(1 for line in f_cnt if line.strip())

        print(f"📊 총 {total_lines:,}개 행 처리 예정\n")

        BATCH_SIZE = 8
        processed_count = 0

        with (
            open(raw_filepath, "r", encoding="utf-8") as f_in,
            open(embedded_filepath, "w", encoding="utf-8") as f_out,
        ):

            batch_texts = []
            batch_rows = []

            for line in tqdm(f_in, total=total_lines, desc="1024차원 임베딩 생성 중"):
                if not line.strip():
                    continue
                row = json.loads(line)

                # DB 모델 필드명과 일치하는 키로 임베딩용 텍스트 조합
                text_for_embedding = (
                    f"{row.get('section_title', '')} "
                    f"{row.get('article_no', '')} "
                    f"{row.get('content', '')}"
                )
                batch_texts.append(text_for_embedding)
                batch_rows.append(row)

                if len(batch_texts) >= BATCH_SIZE:
                    embeddings = get_embeddings_qwen(batch_texts)

                    for obj, emb in zip(batch_rows, embeddings):
                        obj["embedding"] = emb  # 기존 필드 유지 + embedding 추가
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    processed_count += len(batch_texts)
                    f_out.flush()
                    batch_texts = []
                    batch_rows = []

                    # MPS 메모리 누적 방지
                    if device == "mps":
                        torch.mps.empty_cache()

            # 남은 데이터 처리
            if batch_texts:
                embeddings = get_embeddings_qwen(batch_texts)
                for obj, emb in zip(batch_rows, embeddings):
                    obj["embedding"] = emb
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                processed_count += len(batch_texts)
                f_out.flush()

        print(f"\n✅ 임베딩 완료: {os.path.basename(embedded_filepath)}")
        print(f"   🔢 처리된 행: {processed_count:,}개")

    print(f"\n{'=' * 60}")
    print("🎉 모든 Raw 파일의 1024차원 임베딩 생성이 완료되었습니다!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_embed_pipeline()
