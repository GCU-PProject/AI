"""
Qwen3-Embedding-0.6B 로컬 임베딩 추출 테스트 스크립트
─────────────────────────────────────────────────────────────
이 스크립트는 Hugging Face transformers 라이브러리를 이용하여
Qwen3-Embedding-0.6B 모델을 로컬(MPS/CUDA)에 올리고
텍스트를 1024차원의 벡터로 변환하는 방법을 보여줍니다.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1. 실행 하드웨어 디바이스 설정
# 맥북 GPU(mps)가 있으면 사용하고, NVIDIA GPU(cuda)가 있으면 사용하며, 없으면 CPU를 씁니다.
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🖥️  활성화된 하드웨어 가속기: [{device.upper()}]")

# 2. 로컬에 받아둔 Qwen3-Embedding-0.6B 모델 및 토크나이저 로드
# 로컬 폴더에 다운로드 받아두셨다면 경로를 지정하시면 되고, 
# "Qwen/Qwen3-Embedding-0.6B"를 적으면 허깅페이스 캐시 폴더에서 자동으로 로드합니다.
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

print(f"🔄 {MODEL_ID} 모델 로딩 중 (VRAM 적재)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
).to(device)
print("✅ 모델 로드 성공!")

# 3. 임베딩할 샘플 텍스트와 지시어(Instruction) 준비
# Qwen3-Embedding은 지시어(Instruction) 기반 모델입니다.
# - RAG의 DB에 저장할 문서(Document): "Represent this passage for retrieval: " 를 접두사로 붙여야 함.
# - 유저의 검색 질문(Query): "Represent this query for retrieving relevant documents: " 를 접두사로 붙여야 함.
documents = [
    "It is unlawful for any person who is under the influence of any alcoholic beverage to drive a vehicle.",
    "The landlord shall return the security deposit to the tenant within 21 days after the tenant has vacated the premises."
]

# 문서용 지시어 추가
instruction = "Represent this passage for retrieval: "
queries = [instruction + doc for doc in documents]

# 4. 토큰화 및 GPU 데이터 전송
print("\n🔄 텍스트 토큰화 진행 중...")
encoded_input = tokenizer(
    queries, 
    padding=True, 
    truncation=True, 
    max_length=8192,  # 모델 최대 스펙은 32K이지만, 일반 문서는 8K로 충분
    return_tensors='pt'
).to(device)

# 5. 임베딩 연산 수행 (추론 모드)
print("⚡ GPU/MPS 임베딩 벡터 추출 연산 중...")
with torch.no_grad():
    model_output = model(**encoded_input)
    
    # Qwen3-Embedding은 Decoder-Only 아키텍처이므로 Last Token(마지막 토큰)의 hidden_state를 사용하거나
    # Attention Mask를 고려하여 평균(Mean Pooling) 또는 첫 토큰(EOS/BOS)을 취합니다.
    # 허깅페이스 레시피에 맞게 Attention Mask 기반의 Mean Pooling을 수행합니다.
    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # 1024차원 원본 벡터 획득
    embeddings = sum_embeddings / sum_mask
    
    # Cosine Similarity 비교를 위해 L2 정규화(Normalization) 적용
    embeddings = F.normalize(embeddings, p=2, dim=1)

# 6. 결과 확인
print("\n" + "=" * 60)
print("📊 임베딩 결과 요약")
print("=" * 60)
print(f"• 임베딩 완료된 문장 수: {embeddings.shape[0]}개")
print(f"• 벡터의 차원수 (Dimension): {embeddings.shape[1]}차원")
print("-" * 60)

# 첫 번째 문장의 앞부분 5개 차원만 샘플 출력
for i, doc in enumerate(documents):
    vector_sample = embeddings[i][:5].tolist()
    vector_sample_rounded = [round(v, 4) for v in vector_sample]
    print(f"📄 문장 {i+1}: {doc[:50]}...")
    print(f"🎯 1024차원 중 앞 5차원 샘플: {vector_sample_rounded} ...\n")

print("=" * 60 + "\n")
