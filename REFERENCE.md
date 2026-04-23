# Advanced RAG Architecture Reference Guide

이 문서는 Hugging Face의 [한국어 Advanced RAG 구현 쿡북](https://huggingface.co/learn/cookbook/ko/advanced_ko_rag)을 레퍼런스로 하여 작성되었습니다. 
GLAW AI 프로젝트의 현재 개발 상태를 점검하고, 추후 발표 자료(PPT) 및 `README.md` 작성 시 아키텍처 설명의 뼈대로 활용하기 위한 문서입니다.

---

## 1. RAG의 발전 단계와 GLAW AI의 현위치

RAG(Retrieval-Augmented Generation)는 발전 단계에 따라 세 가지로 분류됩니다.

1. **Naive RAG**: 가장 기본적인 형태로 `질문 ➡️ 검색 ➡️ 결합 ➡️ LLM 생성`의 단순한 파이프라인.
2. **Advanced RAG**: Naive RAG의 한계를 극복하기 위해 **질문 전처리(Pre-retrieval)**와 **검색 후처리(Post-retrieval)** 등 고급 기법을 추가한 형태.
3. **Modular RAG**: 시스템을 여러 독립 모듈로 나누어 필요에 따라 검색기, 프롬프트, 도구를 동적으로 선택하는 가장 유연한 형태.

💡 **GLAW AI의 현위치: Advanced RAG**
GLAW AI는 단순 검색을 넘어 다음과 같은 최적화 기법을 적용한 **Advanced RAG** 단계에 도달해 있습니다.
- **검색 전처리**: 과거 대화 내역 기반의 **질문 재구성(Query Reformulation)** 및 한-영 **질문 번역(Query Translation)**.
- **검색 최적화**: 거리 기반 임계값(Threshold) 필터링 및 확장된 컨텍스트(Top-K=7) 활용.
- **생성 가드레일**: 속인주의 등 타국 법률 충돌에 대비한 정교한 프롬프트 튜닝.

---

## 2. RAG 구현 핵심 7단계 및 GLAW AI 매핑

쿡북에서 제시하는 RAG 구현 7단계와, GLAW AI가 이를 어떻게 프로젝트에 적용했는지 매핑한 표입니다. 발표 시 이 구조를 따르면 매우 전문적으로 보일 수 있습니다.

### Step 1. 문서 불러오기 (Document Loading)
- **일반적 방식**: LangChain의 `PyPDFLoader`, `WebBaseLoader` 등을 활용해 정형/비정형 텍스트 로드.
- **GLAW AI 적용**: 캘리포니아, 호주 등 각국 정부의 공식 법률 사이트 데이터를 크롤링하고 파이썬 스크립트(`data_process_*.py`)를 통해 구조화된 데이터(JSON/CSV)로 정제하여 로드.

### Step 2. 문서 나누기 (Chunking/Splitting)
- **일반적 방식**: `CharacterTextSplitter` 등을 사용하여 글자 수(1000자 등) 단위로 기계적으로 자름.
- **GLAW AI 적용 (Custom Splitting)**: 정규표현식을 활용하여 실제 법률의 **조항(Article/Section)** 단위로 정교하게 커스텀 청킹(Chunking). 문맥이 중간에 끊기는 현상을 원천 차단함.

### Step 3. 임베딩 및 벡터 DB 저장 (Embedding & Storage)
- **일반적 방식**: FAISS, Chroma 등의 로컬/인메모리 벡터 저장소 사용.
- **GLAW AI 적용**: GCP Vertex AI의 `text-embedding-005` 모델을 사용하여 고차원 벡터로 변환 후, 프로덕션 레벨의 **PostgreSQL (`pgvector` 확장)** 에 저장. 메타데이터(국가, 법률 종류) 필터링이 가능하도록 관계형 DB의 장점을 결합함.

### Step 4. 검색기 만들기 (Retrieval)
- **일반적 방식**: 기본 검색, MMR(다양성 확보), 하이브리드 검색(Ensemble), Reranker(재순위화) 등.
- **GLAW AI 적용**: 벡터 기반 유사도 검색(Similarity Search)을 기반으로 하되, `MAX_DISTANCE_THRESHOLD`로 무관한 문서를 쳐내고 `TOP_K=7`로 검색 범위를 확장하여 "Lost in the middle" 현상과 "검색 누락" 문제를 동시에 해결함.
- *(Next Step 예정)*: 하이브리드 검색(BM25 결합) 또는 Reranker(Cross-encoder) 도입 고려.

### Step 5. 프롬프트 준비하기 (Prompting)
- **일반적 방식**: 모델에 맞는 템플릿(예: Llama, OpenAI 템플릿) 설정 및 역할 부여.
- **GLAW AI 적용**: YAML 파일(`chat.yaml`)로 프롬프트를 분리 및 관리. 답변의 구조화(핵심 요약, 상세 내용, 주의사항)를 강제하고, "환각 방지"와 "해외 여행객 보호(연방법, 속인주의 경고)"라는 상충하는 목표를 완벽히 조율해 낸 프롬프트 엔지니어링 달성.

### Step 6. 체인 구성하기 (Chain)
- **일반적 방식**: LangChain의 LCEL 문법(`Prompt | LLM | Parser`)으로 파이프라인 구성.
- **GLAW AI 적용**: 번역 로직 ➡️ 검색 로직 ➡️ LCEL 체인으로 이어지는 멀티 스텝 비동기 파이프라인(`chat_service.py`) 구축 완료.

### Step 7. 질문하고 답변 받기 (Generation & Streaming)
- **일반적 방식**: `invoke`를 통한 단일 답변 출력.
- **GLAW AI 적용**: Gemini 2.5 Flash 모델을 활용하여 신속하고 일관성 있는 한국어 답변을 도출하며, FastAPI `StreamingResponse` (SSE)를 통해 실시간 타이핑 효과(Streaming)를 프론트엔드에 제공.

---

## 3. 요약 및 향후 액션 아이템

위의 7단계를 기반으로 보았을 때, GLAW AI는 데이터 파이프라인부터 서빙 단계까지 상당히 수준 높은 구현이 완료되었습니다. 
발표 자료 작성 시 위 매핑 내용을 바탕으로 **"단순한 AI API 호출이 아닌, 법률 도메인에 특화된 데이터 처리와 사용자 보호를 위한 Advanced RAG 설계"**를 강조할 수 있습니다.
