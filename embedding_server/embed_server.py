"""
Qwen3-Embedding-0.6B 임베딩 서버 (HTTP API)
───────────────────────────────────────────────────────────────────
[역할]
  임베딩 전용 VM에서 uvicorn으로 실행되며, 텍스트를 받아 1024차원 벡터를 반환한다.
  실행: uvicorn embed_server:app --host 0.0.0.0 --port 8081

[중요 — Qwen3 공식 가이드 준수 / 벡터 정합성]
  이 서버는 '질문(query)' 임베딩 전용이다.
  - 질문에는 Instruct 형식("Instruct: {task}\nQuery: {query}")을 적용한다. (공식 권장)
  - 문서(passage)는 data_embed_to_file_qwen.py에서 '접두사 없이' 임베딩한다.
  - pooling 방식(last-token)과 L2 정규화는 문서/질문이 동일해야 한다.
  ※ Qwen3는 query에만 instruct를 적용하고 document에는 적용하지 않도록 학습되었다.

[엔드포인트]
  GET  /health        → 헬스체크
  POST /embed         → {"inputs": "텍스트" | ["텍스트", ...]} → [[float, ...], ...]
"""

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# 임베딩 전용 VM은 GPU가 없으므로 CPU + float32
DEVICE = "cpu"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3 공식 query instruct 형식: "Instruct: {task}\nQuery: {query}"
# task 설명은 영어로 작성(공식 권장). 우리 서비스(법률 질문→관련 법조항 검색)에 맞춰 정의.
INSTRUCT_TASK = (
    "Given a user's legal question, retrieve relevant legal provisions that answer it"
)


def format_query(query: str) -> str:
    return f"Instruct: {INSTRUCT_TASK}\nQuery:{query}"


print(f"🔄 {MODEL_ID} 로딩 중 ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
print("✅ 모델 로딩 완료")

app = FastAPI(title="Qwen3 Embedding Server", version="2.0.0")


class EmbedRequest(BaseModel):
    inputs: str | list[str]


def _last_token_pool(last_hidden_states, attention_mask):
    """Last-token pooling (Qwen3-Embedding 공식 권장). 좌/우 패딩 모두 처리."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_idx, seq_lengths]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """질문(query) 배치를 Instruct 형식으로 감싸 1024차원 임베딩을 추출한다.

    data_embed_to_file_qwen.py(문서)와 pooling(last-token)·정규화는 동일하고,
    질문에만 Instruct 접두를 적용하는 점이 다르다. (Qwen3 공식 비대칭 방식)
    """
    prepared_texts = [format_query(t) for t in texts]

    encoded = tokenizer(
        prepared_texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output[0]
        attention_mask = encoded["attention_mask"]

        # Last-token pooling
        embeddings = _last_token_pool(token_embeddings, attention_mask)

        # L2 정규화 (Cosine Similarity 호환)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().float().numpy().tolist()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": 1024}


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    return get_embeddings(texts)
