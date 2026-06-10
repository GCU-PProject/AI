# 🌍 GLAW AI - 프로젝트 개발 진행 보고서

> 글로벌 법률 비교 AI 서비스 (RAG 기반)  
> 최종 수정일: 2026-06-01

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

GLAW(Global Law) AI는 전 세계 법률 정보를 AI로 검색하고 비교할 수 있는 서비스이다. 사용자가 법률 관련 질문을 입력하면, RAG(Retrieval-Augmented Generation) 기술을 활용하여 관련 법률 조항을 벡터 검색으로 찾고, Google Vertex AI(Gemini)를 통해 자연어 답변을 생성한다.

### 1.2 주요 기능

| 기능 | 설명 |
|------|------|
| **법률 Q&A** | 특정 국가의 법률에 대해 질문하면 관련 조항을 검색하여 AI 답변 생성 |
| **법률 비교** | 두 국가 간 동일 주제의 법률을 비교하여 공통점과 차이점 분석 |
| **리스크 카드** | 여행자 조건(국가, 목적, 비자, 연령)에 맞는 법적 리스크 카드 조회 |

### 1.3 기술 스택

| 구분 | 기술 | 버전/모델 |
|------|------|-----------|
| API 서버 | FastAPI + Uvicorn | Python 3.13 |
| 데이터베이스 | PostgreSQL + pgvector | GCP Cloud SQL |
| ORM | SQLAlchemy (Async) | asyncpg 드라이버 |
| AI 프레임워크 | LangChain | LCEL 파이프라인 |
| 임베딩 모델 | Qwen3-Embedding-0.6B (자체 호스팅) | 1024차원 |
| 생성 모델 (LLM) | Google Gemini | gemini-3.5-flash |
| 데이터 수집 | Requests, BeautifulSoup4 | 캘리포니아 법률 크롤링 |
| 인프라 | GCP (Cloud SQL, VM, Bastion Host) | SSH 터널링 |

### 1.4 모델 선정 기준

본 프로젝트의 핵심 생성 모델(LLM)로 **`gemini-3.5-flash`**를 채택하였다. 모델 선정 과정에서 다음 요소들을 종합적으로 고려하였다.

1. **안정적인 정식 배포 버전 사용**
   - 테스트용으로 제공되는 프리뷰 버전(AI Studio) 대신, 실제 서비스 환경에서 안정적으로 운영할 수 있는 구글 클라우드(Vertex AI)의 정식 배포 버전을 사용하였다.

2. **객관적인 벤치마크 지표 기반**
   - 법률 AI 전문 벤치마크인 [LegalBench(vals.ai)](https://www.vals.ai/benchmarks/legal_bench)를 참고하였다.
   - 현재 정식 사용 가능한 모델 중, 정확도와 응답 속도, 토큰 당 비용을 모두 고려했을 때 `gemini-3.5-flash`가 가장 합리적인 성능을 보여 최종 채택하였다.

3. **임베딩 모델 — Qwen3-Embedding-0.6B 자체 호스팅**
   - 임베딩 모델은 MTEB 법률 retrieval 벤치마크(`mteb_legal_retrieval_leaderboard.csv`)를 직접 측정하여 선정하였다. `Qwen3-Embedding-0.6B`(0.8677)가 `text-embedding-005`(0.4464) 대비 법률 도메인 정확도가 약 2배 높아, 외부 API 대신 전용 VM에서 직접 서빙(자체 호스팅, 1024차원)하기로 결정하였다. 상세 내용은 `INFRA_DECISION.md` 참고.

---

## 2. Phase 1 - 기획 및 데이터 조사

### 2.1 대상 국가 선정 배경

프로젝트의 첫 단계로 어떤 국가/지역의 법률 데이터를 우선 구축할 것인지 결정하였다. 다음 두 가지 통계 자료를 근거로 **캘리포니아주**를 1순위로 선정하였다.

**유학생 통계 (Open Doors, 2024)**
- Open Doors는 미국 국무부의 후원과 미국 정부의 자금 지원을 받아 미국 내 유학생에 대한 포괄적 통계와 데이터 보고서를 제공한다.
- 통계에 따르면 2024년 유학생이 가장 많은 주는 **1위 캘리포니아, 2위 뉴욕**이다.

**여행객 통계 (미국 관광청 NTTO)**
- 미국 관광청(NTTO)에서 발표한 자료에 따르면 해외 여행객이 가장 많은 주는 **뉴욕, 플로리다, 캘리포니아**이다.
- 출처: https://www.trade.gov/us-states-cities-visited-overseas-travelers

두 통계 모두에서 캘리포니아가 상위에 위치하므로, 해외 체류자와 여행객이 가장 많이 접하게 될 법률 정보로서 캘리포니아 법령을 최우선 구축 대상으로 선정하였다.

### 2.2 법률 데이터 수집의 어려움

초기에는 전 세계 법률을 통합적으로 관리하는 공개 API가 존재할 것으로 기대하였다. 그러나 조사 결과 다음과 같은 어려움이 발견되었다.

- **통합 API 부재**: 전 세계 법률을 통합 제공하는 API는 존재하지 않았다.
- **주 별 개별 관리**: 미국의 경우 각 주마다 법령이 별도로 존재하며, 법률을 조회할 수 있는 공식 사이트도 주마다 다르다.
- **형식 불일치**: 법률 데이터의 HTML 구조, 조항 표기 방식, 분류 체계가 주마다 제각각이라 범용 크롤러 개발이 불가능하였다.
- **RAG 데이터 구축 난이도**: 법률 텍스트는 조항이 매우 길거나 짧은 등 길이 편차가 크고, 조항 간 상호 참조가 많아 단순 청크 분할이 적합하지 않았다.

이러한 이유로 캘리포니아 전용 크롤러를 별도 개발하는 방식을 채택하였다.

---

## 3. Phase 2 - 데이터 파이프라인 구축

법률 원문 데이터를 수집하고, AI가 검색할 수 있는 벡터 데이터로 변환하는 파이프라인을 구축하였다. 전체 파이프라인은 4단계로 구성된다.

```
[크롤링] → [가공] → [임베딩] → [DB 적재]
```

### 3.1 크롤링 및 가공 (`data_crawl_us_ca.py`)

캘리포니아 주 공식 법률 사이트에서 법령 원문을 수집하고, RAG에 적합한 형태로 가공하였다.

- 법률 코드별 크롤링: VEH(차량법), PEN(형법), CIV(민법), EDC(교육법), LAB(노동법), HSC(보건안전법), GOV(정부법), INS(보험법) 등
- HTML에서 법률 조항을 파싱하여 조항 단위로 분리

- 메타데이터 추출: 법률 종류(`law_type`), 목차(`section_title`), 조항 번호(`article_no`), 본문(`content`), 출처 URL(`source_url`)
- 출력: `.jsonl` 형식의 정제된 데이터 파일

### 3.2 임베딩 생성 (`data_embed_to_file.py`)

정제된 법률 텍스트를 벡터(숫자 배열)로 변환하였다.

- **사용 모델**: `Qwen3-Embedding-0.6B` (자체 호스팅)
- **벡터 차원**: 1024차원
- 각 법률 조항의 `content` 필드를 입력으로 하여 1024차원의 임베딩 벡터 생성
- instruction 접두사(`"Represent this passage for retrieval: "`) + mean pooling + L2 정규화 적용
- 쿼리 임베딩(`embed_server.py`)과 동일한 로직을 사용하여 벡터 정합성 보장
- 출력: 임베딩 벡터가 포함된 `.jsonl` 파일

### 3.3 DB 적재 (`db_load_law_data.py`)

생성된 임베딩 데이터를 PostgreSQL 데이터베이스에 저장하였다.

- **데이터베이스**: GCP Cloud SQL (PostgreSQL)
- **벡터 검색 확장**: pgvector 확장 모듈 사용
- **테이블 구조**: `laws` 테이블에 법률 메타데이터와 1024차원 임베딩 벡터를 함께 저장
- **접근 방식**: Bastion Host를 통한 SSH 터널링으로 보안 접근

### 3.4 데이터베이스 스키마

#### `countries` 테이블 (국가 및 지역 정보)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country_id` | BigInteger (PK) | 국가/지역 고유 ID |
| `country_code` | String(10) | 국가 코드 (예: US, GB) |
| `country_name` | String(100) | 국가명 |
| `state_code` | String(10) | 주/지역 코드 (예: CA, NY) |
| `state_name` | String(100) | 주/지역명 |
| `created_at` | DateTime(timezone=True) | 데이터 생성 시각 |
| `updated_at` | DateTime(timezone=True) | 데이터 수정 시각 |

#### `laws` 테이블 (법률 데이터)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `law_id` | BigInteger (PK) | 법률 고유 ID |
| `country_id` | BigInteger (FK) | 국가 ID (countries 테이블 참조) |
| `law_type` | String(20) | 법률 종류 (예: VEH, PEN, CIV) |
| `section_title` | String | 목차/카테고리 (예: CHAPTER 1. Reports) |
| `article_no` | String | 조항 번호 (예: 23152.) |
| `content` | Text | 법률 본문 |
| `source_url` | String | 출처 URL |
| `enactment_date` | DateTime | 제정일 |
| `amendment_date` | DateTime | 개정일 |
| `created_at` | DateTime(timezone=True) | 데이터 생성 시각 |
| `updated_at` | DateTime(timezone=True) | 데이터 수정 시각 |
| `embedding` | Vector(1024) | 임베딩 벡터 (Qwen3-Embedding-0.6B) |

---

## 4. Phase 3 - v1 구현 (RAG 직접 구현)

### 4.1 구현 방식

첫 번째 버전(v1)은 LangChain 등의 프레임워크를 사용하지 않고, Google Vertex AI SDK를 직접 호출하여 RAG 파이프라인을 수동으로 구현하였다.

### 4.2 RAG 처리 흐름 (법률 Q&A)

```
사용자 질문 입력
    ↓
1. 질문 텍스트를 Qwen3-Embedding-0.6B(자체 호스팅)로 벡터 변환
    ↓
2. pgvector의 L2 거리(유클리드 거리) 기반 유사도 검색
   - 조건: 지정된 country_id에 해당하는 법률만 검색
   - 상위 TOP_K(3)개 문서 반환
   - 임계값(MAX_DISTANCE_THRESHOLD = 0.85) 이하인 문서만 유효로 판단
    ↓
3. 검색된 법률 조항을 f-string으로 프롬프트에 삽입
    ↓
4. Google Gemini에 프롬프트 전달하여 답변 생성
    ↓
5. API 응답 반환 (answer + related_law_id_list)
```

### 4.3 주요 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `TOP_K` | 3 | 벡터 검색에서 반환할 최대 문서 수 |
| `MAX_DISTANCE_THRESHOLD` | 0.90 | L2 거리 기준 임계값. 이 값 이하인 문서만 유효한 검색 결과로 사용 |
| `temperature` | 0 | LLM의 무작위성을 최소화하여 일관된 답변 생성 |
| `max_output_tokens` | 4096 (Q&A) / 2048 (비교) | 답변 최대 길이 |

### 4.4 API 엔드포인트

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/api/qna` | 법률 Q&A (단일 국가) |
| POST | `/api/compare` | 법률 비교 (두 국가) |
| POST | `/api/risk` | 리스크 카드 조회 |

### 4.5 v1의 한계점

- **유지보수 어려움**: 임베딩, 검색, 프롬프트, LLM 호출이 모두 하나의 함수 안에 수동으로 연결되어 있어 수정 시 전체 흐름을 파악해야 함
- **확장성 부족**: 대화 기록(Memory), Reranker, Streaming 등 추가 기능 구현 시 기존 코드를 대폭 수정해야 함
- **질문 언어 불일치**: 한국어 질문을 영어 법률 데이터와 직접 비교하여 벡터 거리 계산 정확도가 떨어짐

---

## 5. Phase 4 - v2 구현 (LangChain 리팩토링) ✅ 현재 완료

### 5.1 리팩토링 목표

v1의 수동 구현 방식을 LangChain 프레임워크로 전환하여 다음 목표를 달성하였다.

- **모듈화**: 각 단계(임베딩, 검색, 프롬프트, LLM)를 독립적인 LangChain 컴포넌트로 분리
- **확장성 확보**: LangChain의 Memory, Reranker, Streaming 등을 쉽게 추가할 수 있는 구조
- **코드 가독성 향상**: LCEL(LangChain Expression Language) 파이프라인으로 데이터 흐름을 명확하게 표현

### 5.2 v1 → v2 컴포넌트 대응표

> ※ 아래 표는 LangChain 리팩토링 당시(v2) 시점의 기록이다. 임베딩은 이후 Phase에서 `VertexAIEmbeddings` → **자체 호스팅 Qwen3(`RemoteEmbeddings`, HTTP)** 로 교체되었다(`INFRA_DECISION.md` 참고).


| 구분 | v1 (직접 구현) | v2 (LangChain) | 변경 이유 |
|------|---------------|----------------|-----------|
| 임베딩 | `TextEmbeddingModel.from_pretrained()` | `VertexAIEmbeddings` | LangChain 통합 관리, 초기화 1회 |
| LLM | `GenerativeModel` | `ChatVertexAI` | LCEL 체인 호환, 파라미터 선언적 관리 |
| 프롬프트 | Python f-string 직접 조립 | `ChatPromptTemplate` | 변수 치환 자동화, 이중 이스케이프 불필요 |
| 체인 연결 | 각 단계를 수동으로 순차 호출 | LCEL 파이프라인 (`prompt \| llm \| parser`) | 데이터 흐름 명확화 |
| JSON 파싱 | `json.loads()` 수동 파싱 | `JsonOutputParser` | 자동 파싱 + 에러 핸들링 |
| 검색 | 직접 SQLAlchemy 쿼리 | 커스텀 Retriever (SQLAlchemy 래핑) | 기존 DB 스키마 유지하면서 LangChain 호환 |

### 5.3 커스텀 Retriever 설계

LangChain의 기본 벡터 스토어(`PGVector`)는 자체 테이블 구조를 사용한다. 그러나 이미 구축된 `laws` 테이블의 스키마를 변경하지 않기 위해, 기존 SQLAlchemy 쿼리를 그대로 사용하는 **커스텀 Retriever**를 설계하였다.

```python
async def retrieve_laws(query, country_id, db) -> tuple[List[Document], List[int]]:
    # 1. 자체 호스팅 Qwen3 임베딩 서버(RemoteEmbeddings)로 질문 벡터화
    query_vector = embeddings.embed_query(query)

    # 2. 기존 SQLAlchemy 쿼리로 L2 거리 기반 검색
    stmt = select(Law, Law.embedding.l2_distance(query_vector))
           .where(Law.country_id == country_id)
           .order_by(Law.embedding.l2_distance(query_vector))
           .limit(TOP_K)

    # 3. 결과를 LangChain Document 형태로 변환하여 반환
    return documents, law_ids
```

이 방식으로 기존 데이터베이스 스키마를 전혀 수정하지 않으면서도 LangChain 체인에 자연스럽게 편입할 수 있었다.

### 5.4 질문 번역 기능 추가

**문제**: 사용자가 한국어로 질문하면, 영어로 저장된 법률 데이터와의 벡터 거리가 커져 검색 정확도가 떨어진다.

**해결**: 검색 전에 질문을 영어로 번역하는 단계를 추가하였다. 최종 답변 생성 시에는 원본 질문(한국어)을 그대로 사용하여 사용자 언어로 답변을 생성한다.

```
v1: 한국어 질문 → 임베딩 → 검색 → 답변
v2: 한국어 질문 → [영어 번역] → 임베딩 → 검색 → 답변 (원본 질문 사용)
```

번역에는 별도 API가 아닌 기존 LLM(Gemini)을 활용하며, 추가 비용 없이 기존 인프라를 재사용한다.

> **2026-06-10 변경 결정:** 번역 품질을 우선하기 위해 검색용 질문 번역을
> Google Cloud Translation LLM(`general/translation-llm`)으로 분리할 예정이다.
> 외부 평가, Google 공식 가이드, 선택 한계 및 자체 검증 계획은
> [`TRANSLATION_MODEL_DECISION.md`](TRANSLATION_MODEL_DECISION.md)에 정리하였다.

```python
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate the user's message to English. "
               "Output ONLY the translated text, nothing else. "
               "If the message is already in English, return it as-is."),
    ("human", "{query}"),
])

async def translate_query(query: str) -> str:
    chain = translation_prompt | llm | StrOutputParser()
    return await chain.ainvoke({"query": query})
```

### 5.5 프롬프트 엔지니어링

LLM이 근거 없는 답변을 생성하거나(할루시네이션), 일관되지 않은 형식으로 답변하는 것을 방지하기 위해 다음과 같은 가이드라인을 프롬프트에 포함하였다.

**가드레일 (답변 품질 보장)**
1. 반드시 제공된 근거 자료만을 기반으로 답변 작성
2. 근거 자료가 질문과 관련 없으면 "답변 불가" 반환 (할루시네이션 방지)
3. 없는 내용은 절대 지어내지 않음

**답변 형식 통일**
```
- 결론: (1문장으로 명확하게 제시)
- 상세 내용: (법률 조항을 근거로 2~4문장 요약)
- [참고 법령]: VEH 23123, CIV 1950.7(c) 등
- ※ 본 답변은 법률적 조언이 아니며 정보 제공을 목적으로 합니다.
```

**인용 방식 개선**
- 초기: LLM이 내부 DB ID(예: "문서 ID: 68722")를 인용하거나, 본문 중간에 법률 코드를 삽입하는 등 일관되지 않은 인용이 발생
- 개선: 컨텍스트 포맷에서 DB ID를 제거하고 법률 코드를 헤더로 표시. 프롬프트에서 본문 내 법률 코드 삽입을 금지하고, 답변 끝에 [참고 법령] 섹션으로 분리

### 5.6 프로젝트 디렉토리 구조

v2 리팩토링 완료 후, v1 코드를 제거하고 단일 구조로 통합하였다.

```
src/
├── api/                      # API 엔드포인트
│   └── endpoint/
│       ├── chat.py            # 법률 Q&A (/api/qna)
│       ├── compare.py         # 법률 비교 (/api/compare)
│       └── risk.py            # 리스크 카드 (/api/risk)
├── core/
│   ├── config.py              # 환경변수 관리 (pydantic-settings)
│   ├── database.py            # 비동기 DB 연결 (asyncpg)
│   └── models.py              # SQLAlchemy ORM 모델 (Country, Law, Risk, RiskList)
├── schemas/
│   ├── chat.py                # Q&A 요청/응답 스키마
│   ├── compare.py             # 비교 요청/응답 스키마
│   ├── risk.py                # 리스크 요청/응답 스키마
│   └── common.py              # 공통 응답 스키마
├── services/                  # 비즈니스 로직 (LangChain 기반)
│   ├── chat_service.py        # RAG Q&A 파이프라인
│   ├── compare_service.py     # 법률 비교 분석
│   ├── risk_service.py        # 리스크 카드 조회
│   └── memory.py              # 대화 기록 관리 (세션)
├── prompts/                   # LLM 프롬프트 (YAML 외부화)
│   ├── chat.yaml              # Q&A 답변 생성
│   ├── compare.yaml           # 비교 분석
│   ├── translation.yaml       # 질문 번역
│   ├── contextualize.yaml     # 질문 재구성
│   ├── risk_card_topics.yaml  # 리스크 주제 생성 (스크립트용)
│   └── risk_card_content.yaml # 리스크 본문 생성 (스크립트용)
├── scripts/                   # 유틸리티 스크립트
│   ├── data_crawl_us_ca.py      # 캘리포니아 법률 크롤링 및 가공
│   ├── data_process_us_ca.py    # 크롤링 데이터 가공
│   ├── data_process_au.py       # 호주 데이터 가공
│   ├── data_process_ko.py       # 한국 데이터 가공
│   ├── data_embed_to_file.py    # 임베딩 생성
│   ├── data_check_distance.py   # 벡터 거리 테스트
│   ├── data_check_models.py     # GCP 모델 연결 테스트
│   ├── ragas_evaluate_rag.py    # RAG 성능 평가
│   ├── ragas_generate_dataset.py # RAGAS 평가 데이터셋 생성
│   ├── db_load_law_data.py      # DB 적재
│   ├── db_insert_countries.py   # 국가 초기 데이터 삽입
│   ├── db_check_connection.py   # DB 연결 확인
│   ├── db_create_tables.py      # 전체 테이블 생성
│   ├── db_load_risk_cards.py    # 리스크 카드 DB 적재
│   └── data_risk_generate_cards.py   # 리스크 카드 생성 (LLM 호출)
└── main.py                    # FastAPI 앱 진입점
```

### 5.7 API 스펙

#### POST `/api/qna` (법률 Q&A)

**요청:**
```json
{
    "query": "음주운전 처벌 기준이 뭐야?",
    "country_id": 1
}
```

**응답:**
```json
{
    "isSuccess": true,
    "code": "AI200",
    "message": "성공입니다.",
    "result": {
        "answer": "결론: 음주운전 시 혈중알코올농도에 따라...\n[참고 법령]: VEH 23152, VEH 23153\n※ 본 답변은 법률적 조언이 아니며 정보 제공을 목적으로 합니다.",
        "related_law_id_list": [68722, 68724],
        "search_success": true
    }
}
```

#### POST `/api/compare` (법률 비교)

**요청:**
```json
{
    "query": "음주운전 처벌 기준이 뭐야?",
    "country_id_1": 1,
    "country_id_2": 2
}
```

**응답:**
```json
{
    "isSuccess": true,
    "code": "AI200",
    "message": "성공입니다.",
    "result": {
        "search_success": true,
        "country_1_result": {
            "related_law_ids": [68722, 68724],
            "summary": "캘리포니아 법률 요약..."
        },
        "country_2_result": {
            "related_law_ids": [12345],
            "summary": "비교 국가 법률 요약..."
        },
        "compare_summary": {
            "common": "공통점 분석...",
            "diff": "차이점 분석..."
        }
    }
}
```

### 5.8 인프라 및 인증

| 항목 | 로컬 개발 환경 | 운영 환경 (예정) |
|------|---------------|-----------------|
| DB 접근 | SSH 터널링 (Bastion Host 경유) | VPC 내부 직접 접근 |
| GCP 인증 | 서비스 계정 키 파일 (.json) + `load_dotenv()` | VM/Cloud Run 서비스 계정 연결 (키 파일 불필요) |
| 서버 실행 | `uvicorn src.main:app --reload` | GCP VM에 배포 |

---

## 6. Phase 5 - 대화 기록(Memory) 구현 및 RAG 최적화

### 6.1 목표

v2 서비스에 대화 맥락 기억 기능을 추가하여, 사용자가 후속 질문을 할 때 이전 대화를 참고할 수 있도록 개선한다.

### 6.2 구현 내용

| 파일 | 역할 |
|------|------|
| `services/memory.py` (신규) | 대화 기록 저장, 질문 재구성, 기록 저장 |
| `services/chat_service.py` | session_id 파라미터 추가, 메모리 연동 |
| `schemas/chat.py` | ChatRequest에 session_id 필드 추가 (Optional) |
| `api/endpoint/chat.py` | session_id를 서비스에 전달 |

**핵심 로직 - 질문 재구성 (Contextualize Question):**
```
사용자: "그러면 벌금은 얼마야?"
    ↓ [이전 대화 기록 참조]
내부: "캘리포니아 음주운전(DUI)의 벌금은 얼마인가?"  ← 독립적 질문으로 변환
    ↓ [이 질문으로 벡터 검색]
결과: 정확한 법률 조항 검색 성공
```

### 6.3 메모리 저장 방식

현재는 **인메모리(Python 딕셔너리)** 방식으로, 서버 재시작 시 대화 기록이 초기화된다. 실서비스 배포 시 DB 저장 방식으로 전환이 필요하다.

### 6.4 테스트 결과

**테스트 질문:**
1. "음주운전을 하면 처벌이 뭐야?" (1차 질문)
2. "그럼 벌금은 얼마나 나와?" (후속 질문)

| 조건 | 1차 질문 | 2차 후속 질문 | 결과 |
|------|---------|-------------|------|
| session_id 있음 | ✅ 정상 답변 | ✅ 맥락 기억, 음주운전 벌금 답변 | 성공 |
| session_id 없음 | ✅ 정상 답변 | ✅ "답변 어렵다" (맥락 없으니 정상) | 성공 |

**테스트 스크린샷:**

#### ✅ session_id 있음 → 맥락 기억하여 정상 답변
![session_id 있음 - 맥락 기억](docs/images/memory_test_with_session.png)

#### ❌ session_id 없음 → 맥락 없어 답변 불가 (정상 동작)
![session_id 없음 - 맥락 없음](docs/images/memory_test_without_session.png)

### 6.5 RAG 통합 테스트 및 최적화 (2026-03-03)

데이터 적재 완료 후 RAG 전체 파이프라인을 테스트하면서 발견된 문제점들을 수정하였다.

#### 6.5.1 질문 재구성 프롬프트 개선

**문제**: LLM이 질문을 재구성하지 않고, 대신 전체 답변을 생성하는 현상이 발생하였다.

```
# 비정상 동작 예시
질문 재구성: '과속 벌금은 얼마인가요?'
→ '과속 벌금은 제한 속도 위반 정도에 따라 다릅니다...' (답변을 생성해버림)
```

**해결**: 프롬프트에 few-shot 예시를 추가하여 원하는 출력 형식을 명확히 보여주었다.

```python
# 변경 후 프롬프트 (few-shot 예시 포함)
"Your ONLY job is to rewrite the user's latest message as a standalone question."
"Output ONLY the rewritten question. No answers, no explanations, no extra text."
"Examples:"
"- Chat history about DUI, user says '그러면 벌금은?' → '음주운전 벌금은 얼마인가요?'"
"- User says '교통사고 처벌이 뭐야?' (no relevant history) → '교통사고 처벌이 뭐야?'"
```

#### 6.5.2 슬라이딩 윈도우 메모리 적용

**문제**: 같은 세션에서 대화가 길어질수록 오래된 주제가 질문 재구성을 오염시켜, 재구성 품질이 저하되는 현상이 발생하였다.

**해결**: 대화 기록 전체 대신 최근 3쌍(6개 메시지)만 참고하는 슬라이딩 윈도우 방식을 적용하였다.

```python
MEMORY_WINDOW_SIZE = 3  # 최근 3쌍만 참고
recent_messages = history.messages[-(MEMORY_WINDOW_SIZE * 2):]
```

| 윈도우 크기 | 선정 근거 |
|-------------|-----------|
| 3쌍 (6메시지) | LangChain 기본값(5쌍)보다 작지만, 법률 Q&A의 후속 질문은 직전 1~2턴만 참고하면 충분하므로 3쌍으로 설정 |

#### 6.5.3 빈 문자열 안전장치 추가

**문제**: 질문 재구성 결과가 빈 문자열(`''`)로 반환되는 경우, 이후 번역 함수에 빈 입력이 전달되어 `400 Model input cannot be empty` 에러가 발생하였다.

**해결**: 재구성 결과가 비어있거나 공백만 있으면 원본 질문으로 폴백하는 안전장치를 추가하였다.

```python
if not contextualized or not contextualized.strip():
    print(f"⚠️ 질문 재구성 결과가 비어있어 원본 질문을 사용합니다: '{query}'")
    return query
```

#### 6.5.4 검색 임계값 조정 (0.85 → 0.90)

**문제**: 한국어 질문을 영어로 번역할 때, 미묘한 뉘앙스 차이로 인해 관련 법률이 임계값을 아슬아슬하게 넘겨 누락되는 현상이 발생하였다.

```
# 한국어 번역: "What should I do if..." → 검색 결과 3개 (행동 중심 뉘앙스)
# 영어 직접:  "What happens if..."     → 검색 결과 5개 (결과 중심 뉘앙스)
```

**해결**: `MAX_DISTANCE_THRESHOLD`를 0.85에서 0.90으로 완화하였다. TOP_K=5로 최대 문서 수가 제한되어 있으므로, 임계값을 올려도 관련 없는 문서가 대량 유입될 위험이 없다.

#### 6.5.5 최대 출력 토큰 증가 (1024 → 2048)

한국어는 영어보다 토큰을 더 많이 소모하므로, 답변이 중간에 잘리는 현상을 방지하기 위해 최대 출력 토큰을 1024에서 2048로 올렸다.

#### 6.5.6 수정 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `services/memory.py` | 질문 재구성 프롬프트 few-shot 예시 추가 |
| `services/memory.py` | 슬라이딩 윈도우 3쌍(6메시지) 적용 |
| `services/memory.py` | 빈 문자열 안전장치 추가 |
| `services/chat_service.py` | `MAX_DISTANCE_THRESHOLD` 0.85 → 0.90 |
| `services/chat_service.py` | `max_output_tokens` 1024 → 4096 |

### 6.6 입력 검증 강화 (2026-03-05)

사용자의 잘못된 입력이 서비스 내부까지 전달되어 예상치 못한 에러를 발생시키는 것을 방지하기 위해, Pydantic의 검증 기능을 활용하여 스키마 단에서 입력을 검증하도록 구현하였다.

#### 6.6.1 Pydantic Validator란?

Pydantic은 FastAPI에서 요청 데이터를 검증하는 라이브러리이다. `@field_validator`와 `@model_validator` 데코레이터를 사용하면 요청이 들어올 때 자동으로 검증 함수가 실행된다.

```python
# field_validator: 필드 하나를 검증 (예: query가 비어있는지)
@field_validator("query")
def check_query(cls, v):    # cls = 클래스 자체, v = 해당 필드의 값
    if not v or not v.strip():
        raise ValueError("질문을 입력해주세요.")
    return v                 # 통과 시 값을 그대로 반환

# model_validator: 여러 필드를 동시에 검증 (예: 두 국가 ID가 같은지)
@model_validator(mode="after")
def check_country_ids(self):  # self = 모든 필드가 들어있는 객체
    if self.country_id_1 == self.country_id_2:
        raise ValueError("두 국가 ID가 같습니다.")
    return self
```

- `raise ValueError`가 실행되면 서비스 코드까지 도달하지 않고, FastAPI가 자동으로 **422 Validation Error** 응답을 반환한다.
- 스키마에서 검증하면 어떤 엔드포인트를 사용하든 동일한 규칙이 적용된다.

#### 6.6.2 검증 규칙

| 스키마 | 검증 항목 | 방식 | 실패 시 |
|--------|-----------|------|---------|
| `ChatRequest` | 빈 질문 차단 | `@field_validator` | 422 에러 |
| `CompareRequest` | 빈 질문 차단 | `@field_validator` | 422 에러 |
| `CompareRequest` | 같은 국가 비교 방지 | `@model_validator` | 422 에러 |

#### 6.6.3 `field_validator` vs `model_validator` 사용 기준

| 구분 | `field_validator` | `model_validator` |
|------|-------------------|-------------------|
| 검사 대상 | 필드 하나 | 모델 전체 (여러 필드) |
| 인자 | `cls, v` (값 하나) | `self` (객체 전체) |
| return | `return v` (값 하나) | `return self` (객체 전체) |
| 사용 예 | 빈 값 체크, 범위 체크 | 필드 간 비교, 조합 검증 |

#### 6.6.4 country_id 존재 여부 검증

`country_id`가 DB에 존재하는지 확인하는 검증은 **스키마가 아닌 엔드포인트**에서 처리한다. 그 이유는 스키마에서는 DB에 접근할 수 없기 때문이다.

```python
# api/endpoint/chat.py - chat 엔드포인트
country = await db.execute(
    select(Country).where(Country.country_id == request.country_id)
)
if not country.scalar():
    return CommonResponse(code="AI404", message="존재하지 않는 국가 ID입니다")

# api/endpoint/compare.py - compare 엔드포인트
# .in_()을 사용하여 두 국가를 한 번의 쿼리로 동시에 조회
country_results = await db.execute(
    select(Country).where(
        Country.country_id.in_([request.country_id_1, request.country_id_2])
    )
)
countries = country_results.scalars().all()
if len(countries) != 2:  # 2개가 아니면 하나 이상이 존재하지 않는 것
    # 어떤 ID가 없는지 특정하여 에러 메시지에 포함
```

#### 6.6.5 검증 위치별 역할 정리

```
요청 들어옴
    ↓
① 스키마 (schemas/)            → 빈 질문 차단, 같은 국가 비교 차단
    ↓
② 엔드포인트 (api/endpoint/) → country_id DB 존재 여부 체크
    ↓
③ 서비스 (services/)         → 법률 데이터 유무 체크
    ↓
LLM 호출
```

---

### 6.7 비교 서비스 부분 검색 개선 (2026-03-05)

**문제**: 두 국가를 비교할 때, 한쪽에만 데이터가 없으면 양쪽 모두 "자료 없음"으로 반환되는 문제가 있었다.

```python
# 기존 코드: 한쪽이라도 없으면 둘 다 "자료 없음" 반환
if not docs_1 or not docs_2:
    return { 둘 다 "자료 없음" }
```

**해결**: `or`를 `and`로 변경하여, 둘 다 없을 때만 즉시 반환하고, 한쪽만 없을 때는 있는 쪽의 데이터를 보여주도록 개선하였다.

```python
# 변경 후: 둘 다 없을 때만 즉시 반환
if not docs_1 and not docs_2:
    return { 둘 다 "자료 없음" }

# 한쪽만 없을 때: 없는 쪽은 국가명과 함께 안내 메시지 표시
if not docs_1:
    context_1_text = f"{country_name}의 법률 데이터가 없습니다."
else:
    context_1_text = format_docs(docs_1)
```

이 방식으로 사용자가 한 번의 요청으로 최대한 많은 정보를 얻을 수 있게 되었다.

---

### 6.8 에러 핸들링 세분화 (2026-03-05)

#### 6.8.1 번역 함수 에러 처리

`translate_query()` 함수에 try-except를 추가하여, 번역 실패 시 에러를 상위로 전파하도록 구현하였다. 번역이 실패하면 부정확한 한국어 질문으로 검색하는 것보다, 에러를 반환하여 사용자에게 명확히 알리는 것이 법률 서비스에 더 적합하다고 판단하였다.

```python
async def translate_query(query: str) -> str:
    try:
        translated = await chain.ainvoke({"query": query})
        if not translated or not translated.strip():
            raise ValueError("번역 결과가 없습니다.")
        return translated
    except Exception as e:
        print(f"번역 실패: {str(e)}")
        raise  # 에러를 엔드포인트의 except까지 전파
```

#### 6.8.2 엔드포인트 에러 세분화

기존에는 모든 에러를 `except Exception`으로 한 번에 처리하여 "서버 내부 오류"로만 반환하였다. 이를 에러 종류별로 세분화하여 사용자에게 더 명확한 에러 메시지를 제공하도록 개선하였다.

```python
try:
    result_data = await generate_answer(...)

except ConnectionError:         # DB 연결 실패 (SSH 터널 끊김 등)
    return CommonResponse(code="AI503", message="DB 연결 실패")

except ValueError as e:         # 번역 실패, 빈 결과 등
    return CommonResponse(code="AI400", message=f"요청 처리 중 오류: {str(e)}")

except Exception as e:          # 예상 못 한 기타 에러 (최후의 안전망)
    return CommonResponse(code="AI500", message="서버 내부 오류")
```

**except 순서가 중요**: 위에서부터 순서대로 체크하며, `Exception`은 모든 에러의 부모이므로 반드시 마지막에 위치해야 한다. 만약 `except Exception`을 첫 번째로 놓으면 `ConnectionError`와 `ValueError`가 절대 잡히지 않는다.

#### 6.8.3 에러 코드 체계

| 코드 | 의미 | 발생 상황 |
|------|------|-----------|
| AI200 | 성공 | 정상 응답 |
| AI400 | 요청 오류 | 번역 실패, 값 오류 |
| AI404 | 찾을 수 없음 | 존재하지 않는 country_id |
| AI500 | 서버 내부 오류 | 예상 못 한 에러 |
| AI503 | 서비스 불가 | DB 연결 실패 |

---

### 6.9 대화 기록 저장 방식 변경 (2026-03-05)

**문제**: 원본 질문("벌금은?")을 저장하면, 슬라이딩 윈도우(3쌍)에서 핵심 키워드("교통사고")가 사라질 수 있다.

```
# 원본 저장 시 문제점
메시지 기록: ["교통사고 어떻게 해?", "벌금은?", "면허 정지 기간은?"]
→ 다음 질문 시 "교통사고" 키워드가 윈도우에서 사라짐!
```

**해결**: `save_to_history`에서 원본 질문 대신 재구성된 질문을 저장하도록 변경하였다.

```python
# 변경 전
save_to_history(session_id, query, final_answer)           # 원본: "벌금은?"

# 변경 후
save_to_history(session_id, search_query, final_answer)    # 재구성: "교통사고 벌금은 얼마인가?"
```

이렇게 하면 슬라이딩 윈도우에서 핵심 키워드가 계속 유지되어, 대화가 길어져도 맥락 추적이 안정적으로 이루어진다.

---

### 6.10 수정 파일 요약 (2026-03-05)

| 파일 | 변경 내용 |
|------|-----------|
| `schemas/chat.py` | `@field_validator("query")` 추가 - 빈 질문 차단 |
| `schemas/compare.py` | `@field_validator("query")` 추가 - 빈 질문 차단 |
| `schemas/compare.py` | `@model_validator` 추가 - 같은 국가 비교 방지 |
| `api/endpoint/chat.py` | country_id DB 존재 여부 검증 추가 |
| `api/endpoint/chat.py` | 에러 핸들링 세분화 (ConnectionError, ValueError, Exception) |
| `services/chat_service.py` | `translate_query()` 에러 처리 추가 |
| `services/chat_service.py` | `save_to_history` 재구성 질문 저장으로 변경 |
| `services/compare_service.py` | 한쪽만 데이터 없을 때도 비교 결과 반환 |

---

### 6.11 ~~Streaming 응답 구현 (2026-03-09)~~ → 삭제됨 (2026-04-08)

> **참고**: Streaming 기능은 사용하지 않기로 결정하여, 코드베이스 정리 시 제거하였다. 아래는 당시 구현했던 기술적 기록이다.

기존 `/chat` 엔드포인트는 LLM이 전체 답변을 생성한 후 한꺼번에 반환하는 방식이었다. 이 경우 사용자는 답변이 완성될 때까지 빈 화면을 보게 된다. Streaming을 적용하면 LLM이 토큰을 생성하는 즉시 실시간으로 전송하여, 사용자가 첫 응답을 빠르게 받을 수 있다.

#### 6.11.1 기존 방식 vs Streaming 방식

| | 기존 `/chat` | 신규 `/chat/stream` |
|---|---|---|
| 서비스 함수 | `generate_answer` | `generate_answer_stream` |
| LLM 호출 | `chain.ainvoke()` (한꺼번에) | `chain.astream()` (조각씩) |
| 반환 방식 | `return { JSON }` | `yield "data: 조각"` (SSE) |
| 엔드포인트 응답 | `CommonResponse` | `StreamingResponse` |
| 사용자 체감 | 5~10초 대기 후 전체 답변 | 0.5초 만에 첫 줄, 이후 계속 추가 |

#### 6.11.2 핵심 기술: `ainvoke()` vs `astream()`

```python
# 기존: 전체 답변이 완성될 때까지 기다림
final_answer = await chain.ainvoke({"context": context, "question": query})

# Streaming: 토큰이 생성될 때마다 즉시 전송
async for chunk in chain.astream({"context": context, "question": query}):
    yield f"data: {chunk}\n\n"
```

`ainvoke()`와 `astream()`은 LangChain 체인(`prompt | llm | StrOutputParser()`)의 내장 메서드이다. 별도 import 없이 체인 객체에서 바로 사용할 수 있다.

#### 6.11.3 SSE(Server-Sent Events) 형식

Streaming 응답은 SSE 표준 형식을 따른다. 일반 API 응답(JSON)과 달리 `data:` 접두사가 필수이다.

```
data: 음주운전은            ← 답변 텍스트 (프론트에서 화면 표시)
data:  위험합니다.           ← 답변 텍스트 (계속 추가)
data: [META]{"related_law_id_list": [68790], "search_success": true}  ← 메타데이터
data: [DONE]                ← 종료 신호
```

프론트엔드는 이벤트 종류에 따라 분기 처리한다:
- 일반 텍스트 → 화면에 실시간 표시
- `[META]` → 메타데이터(법률 ID 등) 내부 저장
- `[DONE]` → 스트리밍 연결 종료

#### 6.11.4 `yield`와 `return`의 차이

```python
# return: 값 1개를 반환하고 함수 종료
def normal():
    return "전체 답변"    # ← 함수 끝!

# yield: 값을 보내고 계속 진행 (제너레이터)
async def streaming():
    yield "첫 조각"      # ← 보내고 계속!
    yield "둘째 조각"     # ← 또 보내고 계속!
    yield "[DONE]"       # ← 마지막
```

`yield`는 값을 즉시 밖으로 전달하므로 버퍼링(flush) 없이 실시간 전송이 가능하다. FastAPI의 `StreamingResponse`가 `yield`된 값을 받아서 HTTP로 프론트엔드에 전송한다.

#### 6.11.5 엔드포인트 구조

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream_endpoint(request, db):
    return StreamingResponse(
        generate_answer_stream(...),       # yield하는 함수
        media_type="text/event-stream",    # SSE 형식 명시
    )
```

`StreamingResponse`는 FastAPI 내장 응답 클래스로, 제너레이터 함수의 `yield` 값을 실시간으로 HTTP 전송한다. `media_type="text/event-stream"`은 프론트엔드에 "이 응답은 SSE 형식이다"라고 알려주는 역할을 한다.

---

### ~~6.12 수정 파일 요약 (2026-03-09)~~ → 삭제됨 (2026-04-08)

| 파일 | 변경 내용 | 현재 상태 |
|------|-----------|---------|
| `services/chat_service.py` | `generate_answer_stream()` 함수 추가 | ❌ 삭제됨 |
| `api/endpoint/chat.py` | `/chat/stream` 엔드포인트 추가 | ❌ 삭제됨 |

---

### 6.12 RAG 검색 및 가드레일 최적화 (Top-K & 프롬프트 튜닝)

**배경**: GPT-4o mini와 본 서비스(GLAW AI)의 답변 품질을 비교 테스트하는 과정에서, 두 가지 한계점이 발견되어 시스템을 최적화하였다.

#### 1. 아파트 보증금 문제 (Top-K 상향)
- **질문**: "캘리포니아에서 1년 동안 아파트 렌트를 살다가 이사 나왔는데, 집주인이 청소비 명목으로 보증금(Security Deposit)을 한 달째 안 돌려주고 있습니다. 제가 언제까지 보증금을 받을 권리가 있고, 주인이 안 주면 법적으로 어떻게 대응할 수 있나요?"
- **문제 상황**:
  - **GPT-4o mini 답변 원문**:
    > 캘리포니아에서는 보증금 반환 기한이 정해져 있습니다. 일반적으로 집주인은 세입자가 이사 나간 뒤 21일 이내에 보증금 정산을 해야 합니다.
    > 
    > 보통 규칙은 이렇습니다:
    > 21일 이내에 집주인은 돌려줄 보증금 잔액, 공제 내역, 공제 사유, 필요한 경우 영수증/증빙를 제공해야 합니다. 청소비 공제도 가능은 하지만, 실제 필요했고 합리적인 비용이어야 합니다. 단순히 “청소했다”는 이유만으로 과도하게 공제하면 분쟁 대상이 될 수 있습니다.
    > 이미 한 달째 안 돌려준 상태라면:
    > 기한을 넘겼을 가능성이 큽니다. 집주인에게 서면으로 반환 요청을 보내는 것이 좋습니다.
    > (이하 생략 - 소액청구법원, 지역 세입자 지원센터 안내 등)

  - **GLAW AI (최적화 전) 답변 원문**:
    > 죄송합니다. 제공된 정보만으로는 답변하기 어렵습니다.

- **원인 분석**: `TOP_K=3`일 때, 번역 뉘앙스 차이로 상가 보증금, 물건 대여 보증금 등 엉뚱한 조항 3개가 먼저 검색되었고, 정답인 주거용 민법(CIV 1950.5)은 3순위 밖으로 밀려나 LLM에게 전달되지 못함. (없는 사실을 지어내지 않는 환각 방지 로직은 정상 작동함)
- **해결 방안 (`chat_service.py`)**: Gemini Flash 모델의 넓은 컨텍스트 입력창과 저렴한 토큰 비용을 활용하여 `TOP_K`를 3에서 7로 상향.
- **최적화 후 결과**: 
  - **GLAW AI (최적화 후) 답변 원문**:
    > **📌 핵심 요약**
    > 캘리포니아 주거용 부동산 임대차 계약의 경우, 집주인은 세입자가 퇴거한 후 늦어도 21일 이내에 보증금의 남은 부분을 반환하고 명세서를 제공해야 합니다. 집주인이 보증금을 부당하게 보관하거나 반환하지 않을 경우, 실제 손해액 외에 보증금의 최대 두 배에 해당하는 법정 손해배상을 청구할 수 있습니다.
    > 
    > **⚖️ 법적 근거 및 상세 내용**
    > *   **보증금 반환 기한**: 집주인은 세입자가 주거용 부동산에서 퇴거한 후 늦어도 21일 이내에 보증금의 남은 부분을 반환하고, 공제 내역을 명시한 명세서를 제공해야 합니다.
    > *   **공제 가능한 항목**: 집주인은 다음 목적을 위해 합리적으로 필요한 금액만 보증금에서 공제할 수 있습니다.
    >     *   세입자의 임대료 미납에 대한 보상.
    >     *   일반적인 마모 및 손상을 제외하고 세입자 또는 세입자의 손님에 의해 발생한 주택 손상 수리.
    >     *   임대차 종료 시 주택을 임대 시작 당시의 청결 수준으로 되돌리는 데 필요한 청소 비용.
    >     *   임대 계약에 따라 개인 재산 또는 부속물을 복원, 교체 또는 반환해야 하는 세입자의 미래 채무 불이행을 해결하기 위한 비용 (일반적인 마모 및 손상 제외).
    > *   **공제 불가 항목**: 집주인은 다음 사항에 대해 보증금에서 공제하거나 세입자에게 청구할 수 없습니다.
    >     *   임대차 계약 이전에 존재했던 손상이나 결함.
    >     *   일반적인 마모 및 손상.
    >     *   합리적으로 필요하지 않은 전문 카펫 청소 또는 기타 전문 청소 서비스 비용.
    > *   **증빙 자료 제공**: 집주인은 공제된 비용에 대한 증빙 서류(예: 작업 명세서, 영수증, 청구서) 사본을 명세서와 함께 제공해야 합니다.
    > *   **부당한 보증금 보관에 대한 책임**: 집주인이 이 조항을 위반하여 보증금의 전부 또는 일부를 악의적으로 보관하는 경우, 실제 손해액 외에 보증금의 최대 두 배에 해당하는 법정 손해배상 책임을 질 수 있습니다. 법원은 사실 관계에 따라 악의적인 경우 손해배상을 명령할 수 있습니다.
    > *   **소액 재판**: 이와 관련된 소송은 소액 재판 법원에서 진행될 수 있습니다.
    > 
    > **📎 참고 법령**
    > CIV 1950.5 (a), (b), (e), (h), (m), (o)
    > 
    > ※ 본 답변은 법률적 조언이 아니며 정보 제공을 목적으로 합니다. 구체적인 법적 판단은 반드시 전문 변호사와 상담하시기 바랍니다.

#### 2. 대마초 여행객 문제 (프롬프트 가드레일 유연화)
- **질문**: "캘리포니아는 대마초(마리화나)가 합법이라고 들었는데, 저 같은 외국인 여행객(관광비자)이 공원이나 길거리에서 피우거나 한국으로 돌아올 때 조금 챙겨 와도 문제가 없나요?"
- **문제 상황**:
  - **GPT-4o mini 답변 원문**:
    > 아니요. 문제가 됩니다. 캘리포니아에서 대마초가 일부 합법이라도, 외국인 여행객이라고 해서 공원·길거리에서 마음대로 피우거나 한국으로 가져오는 것은 허용되지 않습니다.
    > 
    > 핵심만 말하면:
    > 공원, 길거리, 공공장소에서 흡연은 대부분 불법
    > 캘리포니아에서는 대마초를 공공장소에서 사용하면 안 됩니다. 보통 사유지에서만, 그리고 해당 장소의 규칙을 따라야 합니다.
    > 한국으로 조금이라도 가져가는 건 매우 위험
    > 미국 국내에서 합법이어도, 국경을 넘는 반출·반입은 별개 문제. 한국은 대마초에 매우 엄격해서, 소량이라도 불법 반입으로 큰 문제 될 수 있습니다.
  
  - **GLAW AI (최적화 전) 답변 원문**:
    > **📌 핵심 요약**
    > 캘리포니아 주법에 따르면 21세 이상 성인의 대마초 소지 및 사용은 특정 조건 하에 합법이지만, 공공장소에서의 흡연은 일반적으로 허용되지 않습니다. 또한, 제공된 법률 조항은 대마초를 다른 국가로 반출하는 행위에 대해서는 다루고 있지 않습니다.
    > 
    > **⚖️ 법적 근거 및 상세 내용**
    > *   21세 이상의 성인은 다음을 할 수 있습니다:
    >     *   농축되지 않은 대마초 28.5그램 이하 또는 농축 대마초 8그램 이하를 소지, 가공, 운반, 구매, 취득하거나 보상 없이 다른 21세 이상 성인에게 줄 수 있습니다.
    > (중략)
    > **⚠️ 주의사항 및 예외**
    > *   제공된 법률 조항은 캘리포니아 주 및 지역 법률에 따른 대마초의 합법성을 다루고 있으며, 대마초를 다른 국가로 반출하는 행위에 대해서는 명시적으로 허용하거나 금지하는 내용을 포함하고 있지 않습니다.

- **원인 분석**: "제공된 문서에 없으면 절대 지어내지 마라"는 엄격한 RAG 가드레일이 완벽히 작동한 결과이나, 유저 입장에서는 이를 오독하여 불법 반입을 시도할 수 있는 치명적인 오해(UX 및 안전 리스크)를 유발할 수 있음.
- **해결 방안 (`chat.yaml`)**: 프롬프트 가이드라인 8번에 *"마약류, 비자 등 사용자 본국의 법률(속인주의)과 충돌하여 심각한 불이익이 발생할 소지가 있는 경우, 제공된 조항에 없더라도 일반 상식에 기반하여 가벼운 주의사항을 추가하라"*는 예외 규칙을 삽입.
- **최적화 후 결과**: 
  - **GLAW AI (최적화 후) 답변 원문**:
    > **📌 핵심 요약**
    > 캘리포니아 주법에 따르면 21세 이상의 성인은 특정 제한 내에서 대마초를 소지하고 사용할 수 있지만, 공공장소에서의 흡연 또는 섭취는 일반적으로 허용되지 않습니다. 또한, 캘리포니아 주법은 대마초를 다른 주로 운반하거나 해외로 반출하는 것에 대해 다루지 않으며, 이는 미국 연방법 및 국제법에 저촉될 수 있습니다.
    > 
    > **⚖️ 법적 근거 및 상세 내용**
    > *   **소지 및 사용 허용:** 21세 이상의 성인은 농축되지 않은 대마초는 최대 28.5그램, 농축 대마초는 최대 8그램까지 소지, 운반, 구매, 취득 또는 무상으로 양도할 수 있습니다. 또한 대마초 또는 대마초 제품을 흡연하거나 섭취하는 것이 허용됩니다.
    > *   **공공장소에서의 제한:**
    >     *   공공장소에서 대마초 또는 대마초 제품을 흡연하거나 섭취하는 것은 일반적으로 허용되지 않습니다. (단, 특정 예외는 제공된 조항에 명시된 다른 법률에 따릅니다.)
    >     *   담배 흡연이 금지된 장소에서 대마초를 흡연하는 것도 금지됩니다.
    >     *   어린이가 있는 학교, 보육 시설 또는 청소년 센터 1,000피트 이내에서 대마초를 흡연하는 것은 금지됩니다.
    > *   **차량 내 제한:**
    >     *   운전 중이거나 차량의 승객석에 탑승 중인 경우, 개봉된 대마초 또는 대마초 제품 용기를 소지하거나 대마초를 흡연 또는 섭취하는 것은 금지됩니다.
    > 
    > **⚠️ 주의사항 및 예외**
    > *   **연령 제한:** 대마초 소지 및 사용은 21세 이상에게만 허용됩니다.
    > *   **국제 여행 및 연방법:** 캘리포니아 주법은 대마초를 합법화했지만, 미국 연방법은 여전히 대마초를 불법으로 간주합니다. 따라서 대마초를 주 경계를 넘어 운반하거나 특히 해외로 반출하는 것은 미국 연방법 및 국제법에 따라 심각한 처벌을 받을 수 있습니다. **대한민국 국적자는 속인주의 원칙에 따라 해외에서 대마초 관련 범죄를 저질렀을 경우에도 국내법에 의해 처벌받을 수 있으므로 각별한 주의가 필요합니다.**
    > 
    > **📎 참고 법령**
    > HSC 11362.1, HSC 11362.3, HSC 11362.45, VEH 23220, VEH 23222
    > 
    > ※ 본 답변은 법률적 조언이 아니며 정보 제공을 목적으로 합니다. 구체적인 법적 판단은 반드시 전문 변호사와 상담하시기 바랍니다.

---

## 7. Phase 6 - 평가 시스템 및 다국어 확장 기반 마련

### 7.1 데이터베이스 스키마 고도화 (주/지역 확장)

**배경**: 기존에는 캘리포니아 데이터 임시 구성을 위해 `test_countries`를 사용하였으나, 본격적인 타 주(NY, TX 등) 및 타 국가 법률 확장을 위해 스키마를 고도화하였다.

**변경 내용**:
- `test_countries` 테이블을 `countries`로 정식 변경하고, 미국처럼 주 단위 법률이 존재하는 국가를 위해 `state_code`, `state_name` 컬럼을 추가하였다.
- `laws` 테이블에 법률의 시의성을 판단할 수 있도록 `enactment_date`(제정일)와 `amendment_date`(개정일) 컬럼을 추가하여, 향후 최신 및 유효 법률 우선 검색 기능의 토대를 마련하였다.

### 7.2 RAG 자동 평가(Evaluation) 파이프라인 구축 (RAGAS 도입)

**목표**: 향후 Reranker 도입이나 Hybrid Search 적용 시 검색 품질 변화를 정량적으로 측정하기 위한 자동화된 테스트 베드를 구축한다.

**구현 내용**:
- `src/scripts/ragas_generate_dataset.py` 파이프라인을 구축하여, DB에 적재된 법률 조항(Chunk)을 기반으로 RAG 모델 평가용 합성 데이터셋(Synthetic Dataset)을 일관성 있게 자동 생성한다.
- 최신 **RAGAS 0.4.x API**의 `KnowledgeGraph`와 `Synthesizers` 아키텍처를 도입하여, 단순 검색용 1차원적 문맥 질문(SingleHop)뿐만 아니라 복잡한 추론 질문(MultiHop)까지 다양한 난이도가 포함된 현실적인 평가 데이터셋을 확보할 수 있게 되었다.

---

## 8. 향후 과제 (예정)

| 우선순위 | 항목 | 설명 |
|----------|------|------|
| 높음 | **평가 데이터셋 구축** | RAGAS를 활용해 캘리포니아 법률 기반 Q&A 세트를 생성하고 Baseline 스코어 측정 |
| 높음 | **Reranker 도입** | 벡터 검색 후 크로스 인코더 기반 관련성 재평가를 통한 검색 정확도 향상 |
| 중간 | **Hybrid Search** | 의미 기반(Vector) + 키워드 기반(BM25) 검색 결합 |
| 중간 | **추가 법률 데이터 적재** | 호주, 영국, 뉴욕 등 타 국가/주 정부 법률 데이터 크롤링 및 적재 파이프라인 가동 |
| 낮음 | LangSmith 연동 | LangChain 체인 실행 과정 시각화, 추적 및 지연 시간 분석 |
| 낮음 | 의미적 캐싱 (Cache) | 유사 질문 반복 시 LLM 호출 횟수 감소 (비용 절감) |
| 낮음 | GCP VM 자동화 배포 | CI/CD 구축 및 서버/DB 인프라 프로덕션 수준 확장 |
