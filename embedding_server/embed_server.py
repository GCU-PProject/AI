"""
Qwen3-Embedding-0.6B 임베딩 서버 (HTTP API)
───────────────────────────────────────────────────────────────────
[역할]
  임베딩 전용 VM에서 uvicorn으로 실행되며, 텍스트를 받아 1024차원 벡터를 반환한다.
  실행: uvicorn embed_server:app --host 0.0.0.0 --port 8081

[중요 — 벡터 일치 보장]
  이 파일의 임베딩 로직(접두사 + mean pooling + L2 정규화)은
  data_embed_to_file_qwen.py(데이터 임베딩 스크립트)와 한 글자도 다르지 않아야 한다.
  DB에 저장된 벡터와 같은 방식으로 쿼리를 임베딩해야 검색이 정상 작동한다.

[엔드포인트]
  GET  /health        → 헬스체크
  POST /embed         → {"inputs": "텍스트" | ["텍스트", ...]} → [[float, ...], ...]
"""

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# 임베딩 전용 VM은 GPU가 없으므로 CPU + float32 (오프라인 스크립트와 동일)
DEVICE = "cpu"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# 데이터 임베딩 시 사용한 것과 반드시 동일해야 하는 instruction 접두사
INSTRUCTION = "Represent this passage for retrieval: "

print(f"🔄 {MODEL_ID} 로딩 중 ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
print("✅ 모델 로딩 완료")

app = FastAPI(title="Qwen3 Embedding Server", version="1.0.0")


class EmbedRequest(BaseModel):
    inputs: str | list[str]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """data_embed_to_file_qwen.py의 get_embeddings_qwen()과 동일한 로직."""
    prepared_texts = [INSTRUCTION + t for t in texts]

    encoded = tokenizer(
        prepared_texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)
        attention_mask = encoded["attention_mask"]
        token_embeddings = output[0]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # L2 정규화 (Cosine Similarity 호환)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().tolist()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": 1024}


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    return get_embeddings(texts)
