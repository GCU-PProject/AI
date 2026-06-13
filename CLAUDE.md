# CLAUDE.md — GLaw 프로젝트 사실 정리 (Source of Truth)

> 이 문서는 흩어진 회고(`docs/retrospectives/`)와 평가 데이터(`data/`)에서 **검증된 사실만** 모은 단일 기준 문서다.
> 보고서·포스터·발표 자료의 수치·표현은 모두 이 문서를 기준으로 한다.
> ⚠️ 표시는 문서 간 수치가 어긋나 **확정 전 확인이 필요한 항목**이다.

---

## 1. 프로젝트 개요

- **이름**: GLaw — RAG 기반 다국어 법률 비서 플랫폼
- **팀**: AI 1분반 8조 — 이슬기 · 정소연 · 유호찬 · 최예빈 · 김강호
- **목표**: 해외 체류자(여행·유학·취업)가 낯선 현지 법을 **한국어로 묻고, 실제 법령 조항에 근거한 답변을 출처와 함께** 받도록 하는 서비스
- **핵심 가치**: 일반 LLM의 환각(Hallucination) 한계를 RAG로 보완 — 검색된 법령 조항만 근거로 답변, 출처 제시로 검증 가능

## 2. 시스템 아키텍처

- **클라우드**: GCP, VPC 기반. Nginx(리버스 프록시·퍼블릭) → Spring Boot(백엔드) / FastAPI(AI) → PostgreSQL+pgvector(DB) / 임베딩 전용 VM
- **역할 분리**: Spring = 사용자 인증·온보딩·법률 모아보기·북마크·마이페이지 / FastAPI = AI(QnA·비교·리스크) / 임베딩 VM = Qwen3 자체 호스팅
- **임베딩 서버**: TEI가 아닌 **커스텀 FastAPI 서버**(`embedding_server/embed_server.py`), CPU 추론(float32), `/embed` HTTP 엔드포인트
- **배포**: Terraform(IaC) + GitHub Actions + Docker

## 3. 데이터 — 약 131만 조항 (3개국)

| 국가 | 소스 | 규모 |
|---|---|---|
| 미국 | Federal Statutes, California Code, New York Laws | 약 390,000 조항 |
| 캐나다 | Federal Acts, Ontario, BC Laws | 약 270,000 조항 |
| 호주 | Commonwealth + 7개 주 조례 | 약 640,000 조항 |

- **총계**: 정확히는 **1,313,837건**(`search_db_vector_index.md`). 보고서엔 "약 131만" 또는 "130만 이상"으로. (⚠️ 보고서 본문엔 "130만"으로 적힌 곳 있음 — "131만"으로 통일 권장)
- UK 데이터는 raw만 수집(`data/UK_Law_Data_Raw.jsonl`), **임베딩·서비스엔 미사용** → 보고서엔 3개국만 언급
- **전처리**: 조항/단락 단위 청킹 → Qwen3로 1024차원 벡터화

## 4. RAG 파이프라인 (실제 구현 — `src/services/chat_service.py:generate_answer`)

순서대로:
1. **질문 재구성 (Contextualize)** — `session_id` 있을 때만. 대화 맥락으로 후속 질문을 독립 질문으로 ("그러면 벌금은?" → "캘리포니아 음주운전 벌금은?")
2. **번역 (Translation)** — 한→영, Cloud Translation LLM. 검색에만 사용, **답변 생성엔 원본 한국어 질문** 전달(→ 답변이 한국어로 나옴)
3. **질문 재작성 (Query Rewrite)** — `RAG_ENABLE_REWRITE=True`. 막연한 질문을 검색 친화적 질의로 보강
4. **임베딩 (Embedding)** — Qwen3-Embedding-0.6B, 1024차원
5. **벡터 검색 (Vector Search)** — pgvector, HNSW, top_k=7, L2 threshold 0.95
6. **임계값 필터 (가드레일 ①)** — threshold 초과 조항 제외, 결과 0건이면 LLM 호출 없이 "정보 없음" 반환
7. **답변 생성 (Generation, 가드레일 ②)** — Gemini 3.1 Flash Lite. 프롬프트에 "제공된 조항만 근거" 가드레일 내장. 근거 조항·출처 병기

> **리랭커(Cross-Encoder)는 미도입**. 검토했으나 보류(`search_retrieval_quality_reranker.md`). **향후 과제**로만 언급할 것. 보고서 초안의 "Reranking 단계"는 사실과 다름.

## 5. 모델 & 핵심 기술 결정 (회고 근거)

### 임베딩 — Qwen3-Embedding-0.6B 자체 호스팅
- **선정 근거**: MTEB 영어 법률 retrieval 벤치마크 **평균 NDCG@10** = **0.8677** (232개 모델 중 3위). vs Google `text-embedding-005` **0.4464** → **약 2배**. (`search_embedding_qwen3.md`, `mteb_legal_retrieval_leaderboard.csv`)
  - ⚠️ NDCG@10은 **검색 랭킹 품질** 지표(분류 정확도 아님). 두 모델의 참여 태스크 수가 다름(Qwen3 2개 / 005 3개)이라 완전 동일 태스크 비교는 아님 → "법률 검색 성능 약 2배"로 표현
- **사용 방식**(Qwen3 공식 가이드): 질문에만 Instruct 접두(`Instruct: {task}\nQuery:...`), 문서는 접두 없음, **last-token pooling + L2 정규화**(문서·질문 동일)
- 자체 호스팅 이유: 정확도(범용 API 2배) + 데이터 보안

### 생성 — Gemini 3.1 Flash Lite
- **선정 근거**: 후보(3-flash, 3.1-pro, 3.5-flash, 3.1-flash-lite) RAGAS 비교 후 확정 (`model_thinking_tradeoff.md`, `model_final_selection.md`)
- 법률 서비스는 **Faithfulness(근거충실도) 우선** > FactualCorrectness → lite가 Faithfulness 최고 + **응답 약 2배 빠름**(4.7s vs 9s)
- 추론(thinking)을 끄는 게 RAG에 유리(과한 추론이 컨텍스트 이탈 유발)
- ⚠️ 메모리엔 "gemini-2.1-lite"로 적힌 적 있으나 **모든 docs 기준 정식 명칭은 gemini-3.1-flash-lite**

### 번역 — Cloud Translation LLM (`general/translation-llm`)
- 생성 모델에서 **분리**. 번역 시간 2~7s → **~0.5s** (-76%), 검색 품질 유지 (`model_translation_selection.md`)
- 부수효과: 번역 고정 → 검색 지표(CR/CP)가 생성 모델과 무관해져 실험 통제 개선

### 검색 — PostgreSQL + pgvector + HNSW
- 인덱스: `hnsw (embedding vector_l2_ops)`, **ef_search=100**(최종). top_k=7, L2 threshold **0.95**
- L2 정규화 벡터라 L2 거리 순위 = 코사인 유사도 순위
- ⚠️ **검색 시간 수치 불일치 (보고서 작성 시 주의)** — `search_db_vector_index.md` 기준:
  - 순수 DB 쿼리: 풀스캔 4~12s → HNSW **0.19s**
  - 검색 단계 평균(임베딩 HTTP ~1s 포함): 풀스캔 **5.2s** → ef=40 **1.6s**(재현율↓) → **ef=100(최종) 2.8s**
  - → 포스터/보고서의 "**5.2s → 1.5s**"는 ef=40 기준. **최종(ef=100)은 ~2.8s**. 정직하게 쓰려면 "5.2s → 2.8s (약 2배)" 또는 "DB 쿼리 5초+ → 0.19s" 중 택일

## 6. 평가 (RAGAS)

- **채점 모델 고정**: `gemini-3-flash-preview` (thinking off) — 전 실험 공통 (`eval_ragas_design.md`)
- **지표 4개**: Context Recall(검색 재현율), Context Precision(검색 정밀도), Faithfulness(근거충실도), FactualCorrectness(사실정확도)
- **우선 지표**: 검색=**Recall**(관련 법령 누락 방지 > 정밀도), 생성=**Faithfulness**(근거 검증 가능성 > 사실정확도)
- **테스트셋 2종**: 합성(`ragas_testset_2.csv`, 법령서 자동생성 → 공식적) / **casual**(`ragas_testset_casual.csv`, 실사용자 표현 30문항 수동제작 → 실서비스 반영). **casual이 최종 기준**

### 최종 결과 (casual 테스트셋, 최종 구성 `rag_gemini3.1lite_rewrite_casual_k7_t95`)
| 지표 | RAG 미적용 (순수 LLM) | G.Law (RAG) |
|---|---|---|
| FactualCorrectness | **0.1573** | **0.3433** (≈ 2.2배 향상) |
| Faithfulness | 측정 불가(검색 없음) | **0.8057** |
| Context Recall | 측정 불가 | **0.8222** |
| Context Precision | 측정 불가 | 0.8163 |

- 포스터/보고서 표기: FC **0.16 → 0.34 (약 2배)**, Faithfulness **0.81**, Recall **0.82**
- baseline은 **RAG만 끈 동일 파이프라인**(검색 없이 LLM 단독). "순수 LLM"보다 "**RAG 미적용**"이 정확
- 주의: Faithfulness·Recall은 **baseline과 비교 불가**(검색 없으면 산출 안 됨) → "순수 LLM 대비 2배"는 FC에만 적용
- 참고: 합성 테스트셋에선 Recall 0.95 / Faithfulness 0.76 (수치가 다르니 섞지 말 것)

## 7. AI API (FastAPI) — `docs/API_SPEC.md`

- `POST /api/qna` — query, country_id, session_id → answer, related_law_id_list, search_success
- `POST /api/compare` — query, country_id_1, country_id_2 → 두 국가 요약 + 공통점/차이점
- `POST /api/risk` — country_id, travel_purpose, visa_type, age_band → 위험도 + 대체행동 체크리스트 + 법령 근거
  - 리스크 카드는 조합 단위로 **사전 생성**(`data/risk_cards.json`, 1000 조합 / 약 4,920 카드), 인용 법령 원문과 대조해 자동 팩트체크

## 8. 보고서/포스터용 핵심 수치 요약

| 항목 | 값 |
|---|---|
| 법령 데이터 | 약 131만 조항 (미국 39만 / 캐나다 27만 / 호주 64만) |
| 임베딩 성능 | Qwen3 NDCG@10 0.87 vs text-embedding-005 0.45 (약 2배) |
| FactualCorrectness | RAG 미적용 0.16 → RAG 0.34 (약 2배) |
| Faithfulness | 0.81 (casual) |
| Context Recall | 0.82 (casual) |
| 검색 속도 | 풀스캔 5.28s → 최종 구성(ef=100) 1.36s (`latency_history_30.csv` 측정값) |
| 번역 속도 | 2~7s → ~0.5s |

## 9. 확정 전 확인 필요 (⚠️ 모음)
1. **검색 시간 (확정)**: 최종 구성 행(`rag_gemini3.1lite_rewrite_casual_k7_t95`) 기준 **5.28s → 1.36s**. 2.8s는 3.5flash 실험 행 수치로 우리 최종 구성이 아님. 포스터의 "1.5s"는 ef=40 중간 실험값.
2. **데이터 규모**: "130만" vs "131만" — 통일 (정확값 1,313,837)
3. **모델명**: gemini-3.1-flash-lite로 통일 (메모리의 2.1-lite는 오기)
4. **리랭커**: 미구현 — "향후 과제"로만. 구현된 단계처럼 쓰지 말 것
