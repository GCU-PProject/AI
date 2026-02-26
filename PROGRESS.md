# 🌍 GLAW AI - 프로젝트 개발 진행 보고서

> 글로벌 법률 비교 AI 서비스 (RAG 기반)  
> 최종 수정일: 2026-02-23

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

GLAW(Global Law) AI는 전 세계 법률 정보를 AI로 검색하고 비교할 수 있는 서비스이다. 사용자가 법률 관련 질문을 입력하면, RAG(Retrieval-Augmented Generation) 기술을 활용하여 관련 법률 조항을 벡터 검색으로 찾고, Google Vertex AI(Gemini)를 통해 자연어 답변을 생성한다.

### 1.2 주요 기능

| 기능 | 설명 |
|------|------|
| **법률 Q&A** | 특정 국가의 법률에 대해 질문하면 관련 조항을 검색하여 AI 답변 생성 |
| **법률 비교** | 두 국가 간 동일 주제의 법률을 비교하여 공통점과 차이점 분석 |

### 1.3 기술 스택

| 구분 | 기술 | 버전/모델 |
|------|------|-----------|
| API 서버 | FastAPI + Uvicorn | Python 3.13 |
| 데이터베이스 | PostgreSQL + pgvector | GCP Cloud SQL |
| ORM | SQLAlchemy (Async) | asyncpg 드라이버 |
| AI 프레임워크 | LangChain | v2 구현 |
| 임베딩 모델 | Google Vertex AI | text-embedding-005 (768차원) |
| 생성 모델 (LLM) | Google Gemini | gemini-2.0-flash |
| 데이터 수집 | Requests, BeautifulSoup4 | 캘리포니아 법률 크롤링 |
| 인프라 | GCP (Cloud SQL, VM, Bastion Host) | SSH 터널링 |

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

### 3.1 크롤링 및 가공 (`crawl_us_ca.py`)

캘리포니아 주 공식 법률 사이트에서 법령 원문을 수집하고, RAG에 적합한 형태로 가공하였다.

- 법률 코드별 크롤링: VEH(차량법), PEN(형법), CIV(민법), EDC(교육법), LAB(노동법), HSC(보건안전법), GOV(정부법), INS(보험법) 등
- HTML에서 법률 조항을 파싱하여 조항 단위로 분리

- 메타데이터 추출: 법률 종류(`law_type`), 목차(`section_title`), 조항 번호(`article_no`), 본문(`content`), 출처 URL(`source_url`)
- 출력: `.jsonl` 형식의 정제된 데이터 파일

### 3.2 임베딩 생성 (`embed_to_file.py`)

정제된 법률 텍스트를 벡터(숫자 배열)로 변환하였다.

- **사용 모델**: Google Vertex AI `text-embedding-005`
- **벡터 차원**: 768차원
- 각 법률 조항의 `content` 필드를 입력으로 하여 768차원의 임베딩 벡터 생성
- Vertex AI의 API 호출 제한(Rate Limit)을 고려하여 배치 처리 및 대기 시간 적용
- 출력: 임베딩 벡터가 포함된 `.jsonl` 파일

### 3.3 DB 적재 (`load_to_db.py`)

생성된 임베딩 데이터를 PostgreSQL 데이터베이스에 저장하였다.

- **데이터베이스**: GCP Cloud SQL (PostgreSQL)
- **벡터 검색 확장**: pgvector 확장 모듈 사용
- **테이블 구조**: `laws` 테이블에 법률 메타데이터와 768차원 임베딩 벡터를 함께 저장
- **접근 방식**: Bastion Host를 통한 SSH 터널링으로 보안 접근

### 3.4 데이터베이스 스키마

#### `test_countries` 테이블 (국가 정보)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `country_id` | BigInteger (PK) | 국가 고유 ID |
| `country_code` | String(10) | 국가 코드 (예: US, GB) |
| `country_name` | String(100) | 국가명 |

#### `laws` 테이블 (법률 데이터)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `law_id` | BigInteger (PK) | 법률 고유 ID |
| `country_id` | BigInteger (FK) | 국가 ID |
| `law_type` | String(20) | 법률 종류 (예: VEH, PEN, CIV) |
| `section_title` | String | 목차/카테고리 (예: CHAPTER 1. Reports) |
| `article_no` | String | 조항 번호 (예: 23152.) |
| `content` | Text | 법률 본문 |
| `source_url` | String | 출처 URL |
| `embedding` | Vector(768) | 임베딩 벡터 (text-embedding-005) |

---

## 4. Phase 3 - v1 구현 (RAG 직접 구현)

### 4.1 구현 방식

첫 번째 버전(v1)은 LangChain 등의 프레임워크를 사용하지 않고, Google Vertex AI SDK를 직접 호출하여 RAG 파이프라인을 수동으로 구현하였다.

### 4.2 RAG 처리 흐름 (법률 Q&A)

```
사용자 질문 입력
    ↓
1. 질문 텍스트를 Vertex AI text-embedding-005로 벡터 변환
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
| `MAX_DISTANCE_THRESHOLD` | 0.85 | L2 거리 기준 임계값. 이 값 이하인 문서만 유효한 검색 결과로 사용 |
| `temperature` | 0 | LLM의 무작위성을 최소화하여 일관된 답변 생성 |
| `max_output_tokens` | 1024 (Q&A) / 2048 (비교) | 답변 최대 길이 |

### 4.4 API 엔드포인트

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/api/v1/chat` | 법률 Q&A (단일 국가) |
| POST | `/api/v1/compare` | 법률 비교 (두 국가) |

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
    # 1. LangChain VertexAIEmbeddings로 질문 벡터화
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

### 5.6 프로젝트 디렉토리 구조 변경

v1과 v2를 동시에 유지하기 위해 서비스 폴더를 분리하였다.

```
src/
├── v1_services/              # 기존 직접 구현 (유지)
│   ├── __init__.py
│   ├── chat_service.py       # v1 Q&A 서비스
│   └── compare_service.py    # v1 비교 서비스
├── v2_services/              # LangChain 구현 (신규)
│   ├── __init__.py
│   ├── chat_service.py       # v2 Q&A 서비스 (번역 + LCEL)
│   └── compare_service.py    # v2 비교 서비스 (JsonOutputParser)
├── api/
│   ├── v1/endpoint/          # v1 API 라우터 (/api/v1/chat, /api/v1/compare)
│   └── v2/endpoint/          # v2 API 라우터 (/api/v2/chat, /api/v2/compare)
├── core/
│   ├── config.py             # 환경변수 관리 (pydantic-settings)
│   ├── database.py           # 비동기 DB 연결 (asyncpg)
│   └── models.py             # SQLAlchemy ORM 모델 (Country, Law)
├── schemas/
│   ├── chat.py               # Q&A 요청/응답 스키마
│   ├── compare.py            # 비교 요청/응답 스키마
│   └── common.py             # 공통 응답 스키마
├── scripts/                  # 유틸리티 스크립트
│   ├── crawl_us_ca.py        # 캘리포니아 법률 크롤링 및 가공
│   ├── embed_to_file.py      # 임베딩 생성
│   ├── load_to_db.py         # DB 적재
│   └── check_distance.py     # 벡터 거리 테스트
└── main.py                   # FastAPI 앱 진입점 (v1 + v2 라우터 등록)
```

### 5.7 v2 API 스펙

#### POST `/api/v2/chat` (법률 Q&A)

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

#### POST `/api/v2/compare` (법률 비교)

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

## 6. Phase 5 - 대화 기록(Memory) 구현 (예정)

### 6.1 목표

현재 v2는 각 질문을 독립적으로 처리하며, 이전 질문의 맥락을 기억하지 못한다. LangChain의 Memory 기능을 추가하여 연속 대화가 가능하도록 개선할 예정이다.

**현재:**
```
Q: "음주운전 처벌 기준이 뭐야?" → 답변
Q: "벌금은 구체적으로 얼마야?" → 이전 맥락 모름 → 부정확한 답변
```

**개선 후:**
```
Q: "음주운전 처벌 기준이 뭐야?" → 답변
Q: "벌금은 구체적으로 얼마야?" → 이전 맥락 기억 → 음주운전 벌금 정보 제공
```

### 6.2 사용 예정 기술

- LangChain `ConversationBufferMemory` 또는 `ConversationSummaryMemory`
- 세션 기반 대화 관리 (세션 ID별 대화 기록 분리)

---

## 7. Phase 6 - 추가 고도화 (예정)

| 우선순위 | 항목 | 설명 |
|----------|------|------|
| 높음 | 입력 검증 강화 | 같은 국가 비교 방지, 존재하지 않는 country_id 처리, 빈 질문 차단 |
| 높음 | 에러 핸들링 강화 | 번역 실패 시 원본 질문으로 fallback, DB 연결 끊김 대응 |
| 중간 | Reranker 도입 | 벡터 검색 후 LLM 기반 관련성 재평가로 검색 품질 향상 |
| 중간 | Hybrid Search | 벡터 검색 + 키워드 검색(BM25) 결합 |
| 중간 | Streaming 응답 | LLM 응답을 실시간으로 전달하여 UX 개선 |
| 중간 | 다국어 법률 데이터 확장 | 영국, 싱가포르 등 추가 국가 법률 데이터 구축 |
| 낮음 | LangSmith 연동 | LangChain 체인 실행 과정 시각화 및 디버깅 |
| 낮음 | 캐싱 | 동일 질문 반복 시 LLM 호출 없이 캐시 반환 (비용 절감) |
| 낮음 | 평가 시스템 | RAGAS 등으로 RAG 품질 자동 측정 |
| 낮음 | GCP VM 배포 | 운영 환경 구축 및 서비스 계정 인증 전환 |
