# 🌍 GLAW AI - 글로벌 법률 비교 AI 서비스

> 각국의 법률을 AI로 검색하고 비교할 수 있는 RAG(Retrieval-Augmented Generation) 기반 API 서비스

## 📌 프로젝트 소개

GLAW AI는 사용자의 법률 관련 질문에 대해 관련 법률 조항을 벡터 검색(Vector Search)으로 찾고,
Google Vertex AI(Gemini)를 활용하여 답변을 생성하는 서비스입니다.

### 주요 기능
- **법률 Q&A** (`/api/qna`): 특정 국가의 법률에 대해 질문하면 관련 조항을 검색하여 AI 답변 생성
- **법률 비교** (`/api/compare`): 두 국가 간 법률을 비교하여 공통점과 차이점 분석
- **리스크 카드** (`/api/risk`): 여행자 조건(국가, 목적, 비자, 연령)에 맞는 법적 리스크 카드 조회

### 🗂 국가 ID 매핑 (API 요청 시 참고)

| country_id | 국가 | 주/지역 |
|------------|------|---------|
| 1 | United States (US) | California (CA) |
| 2 | Canada (CA) | - |

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| **Backend** | FastAPI, Uvicorn |
| **Database** | PostgreSQL + pgvector (벡터 검색) |
| **ORM** | SQLAlchemy (Async) |
| **AI/ML** | Google Vertex AI (Gemini, text-embedding-005), LangChain |
| **크롤링** | Requests, BeautifulSoup4 |
| **Lint/Formatter** | Ruff, Black |

## 📁 프로젝트 구조

```
GLAW/AI/
├── src/
│   ├── api/                  # API 엔드포인트
│   │   └── endpoint/
│   │       ├── chat.py            # 법률 Q&A (/api/qna)
│   │       ├── compare.py         # 법률 비교 (/api/compare)
│   │       └── risk.py            # 리스크 카드 (/api/risk)
│   ├── core/                 # 핵심 설정
│   │   ├── config.py              # 환경변수 관리
│   │   ├── database.py            # DB 연결 설정
│   │   └── models.py              # ORM 모델 (Country, Law, Risk, RiskList)
│   ├── schemas/              # 요청/응답 스키마
│   │   ├── common.py              # 공통 응답 포맷 (CommonResponse)
│   │   ├── chat.py                # Q&A 요청/응답
│   │   ├── compare.py             # 비교 요청/응답
│   │   └── risk.py                # 리스크 요청/응답
│   ├── services/             # 비즈니스 로직 (LangChain 기반)
│   │   ├── chat_service.py        # RAG Q&A 파이프라인
│   │   ├── compare_service.py     # 법률 비교 분석
│   │   ├── risk_service.py        # 리스크 카드 조회
│   │   └── memory.py              # 대화 기록 관리 (세션)
│   ├── prompts/              # LLM 프롬프트 (YAML)
│   │   ├── chat.yaml              # Q&A 답변 생성 프롬프트
│   │   ├── compare.yaml           # 비교 분석 프롬프트
│   │   ├── translation.yaml       # 질문 번역 프롬프트
│   │   ├── contextualize.yaml     # 질문 재구성 프롬프트
│   │   ├── risk_card_topics.yaml  # 리스크 주제 생성 프롬프트 (스크립트용)
│   │   └── risk_card_content.yaml # 리스크 본문 생성 프롬프트 (스크립트용)
│   ├── scripts/              # 유틸리티 스크립트
│   │   ├── crawl_us_ca.py         # 미국(캘리포니아) 법률 크롤러
│   │   ├── process_us_ca.py       # 크롤링 데이터 가공
│   │   ├── process_au.py          # 호주 데이터 가공
│   │   ├── embed_to_file.py       # 임베딩 벡터 생성
│   │   ├── load_to_db.py          # DB 적재
│   │   ├── generate_risk_cards.py # 리스크 카드 생성 (LLM 호출)
│   │   ├── load_risk_cards.py     # 리스크 카드 DB 적재
│   │   ├── create_risk_tables.py  # 리스크 테이블 생성
│   │   ├── check_distance.py      # 벡터 거리 테스트
│   │   ├── check_models.py        # GCP 모델 연결 테스트
│   │   ├── insert_countries.py    # 국가 초기 데이터 삽입
│   │   ├── db_check.py            # DB 연결 확인
│   │   ├── evaluate_rag.py        # RAG 성능 평가
│   │   └── generate_dataset.py    # RAGAS 평가 데이터셋 생성
│   └── main.py               # FastAPI 앱 진입점
├── data/                     # 크롤링/임베딩 데이터 (.jsonl)
├── keys/                     # GCP 인증키 (Git 미포함)
├── .env                      # 환경변수 (Git 미포함)
├── requirements.txt          # Python 패키지 목록
└── README.md
```

## ⚙️ 설치 및 실행

### 1. 가상환경 설정
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env.example`을 참고하여 `.env` 파일을 생성합니다.
```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

필수로 확인할 값:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `CORS_ALLOW_ORIGINS` (프론트가 API를 직접 호출하는 경우)

### 3. SSH 터널링 (로컬 개발 시)
DB 서버에 직접 접근할 수 없으므로 SSH 터널링이 필요합니다.
```bash
ssh -i "~/.ssh/gcp_bastion" -N -L 5432:<DB_PRIVATE_IP>:5432 <USER>@<BASTION_IP>
```

### 4. 서버 실행
```bash
uvicorn src.main:app --reload
```

## 📊 데이터 파이프라인

법률 데이터의 수집부터 검색까지의 흐름:

```
1. 크롤링 (crawl_us_ca.py)
   → 법률 원문 수집

2. 가공 (process_us_ca.py)
   → 조항 단위로 분리, 정제

3. 임베딩 (embed_to_file.py)
   → Vertex AI로 텍스트 → 벡터 변환

4. DB 적재 (load_to_db.py)
   → PostgreSQL + pgvector에 저장

5. 검색 (chat_service.py)
   → 질문 임베딩 → L2 거리 기반 유사 법률 검색 → Gemini로 답변 생성
```

## 👥 팀원

| 이름 | 역할 |
|------|------|
| 유호찬 | AI / Data |
| 정소연 | Data |
