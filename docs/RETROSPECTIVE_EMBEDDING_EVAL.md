# 🔧 회고 — 임베딩 방식 개선 & RAG 평가 파이프라인 정비

> 작성일: 2026-06-01
> 주제: BC 데이터 버그 수정 / Qwen3 공식 임베딩 방식 전환 / RAGAS 평가 재설계

---

## 1. 배경

RAG 검색 품질을 개선하고 평가를 재정비하는 과정에서 다음 3가지 문제를 발견하고 수정하였다.

1. 캐나다 BC(British Columbia) 법률 데이터가 DB에 적재되지 않은 문제
2. 임베딩 방식이 Qwen3-Embedding 공식 가이드와 어긋난 문제
3. RAGAS 평가 스크립트가 구버전(이전 모델·단일 국가) 기준으로 동작하던 문제

---

## 2. BC 데이터 누락 버그

### 증상
`laws` 테이블에 `country_id = 6`(British Columbia) 데이터가 0건이었다.
반면 `country_id = 5`(Ontario)에는 정상보다 많은 데이터가 들어가 있었다.

### 원인 — 부분 문자열(in) 매칭 오류
`data_process_ca.py`의 디렉토리 → country_id 분기에서 부분 문자열 매칭(`in`)을 사용했다.

```python
elif "ON" in dir_upper:   # "LEGISLATION-BC"에도 "ON"이 포함됨 (LEGISLATI-ON)
    country_id = 5
elif "BC" in dir_upper:   # 도달하지 못함
    country_id = 6
```

`"LEGISLATION-BC"` 문자열 안의 `LEGISLATION`에 `ON`이 포함되어 있어,
BC 디렉토리가 Ontario(5)로 **오분류**되어 적재되었다.

### 해결
디렉토리명 **접미사 정확 매칭**으로 변경.

```python
if dir_upper.endswith("-FED"):
    country_id = 4
elif dir_upper.endswith("-ON"):
    country_id = 5
elif dir_upper.endswith("-BC"):
    country_id = 6
```

> 데이터 자체는 정상이었고(HuggingFace `a2aj/canadian-laws`에 BC 디렉토리·구조 모두 존재),
> 순수하게 분류 로직 버그였다. → 캐나다 데이터 재처리·재적재 필요.

---

## 3. 임베딩 방식 개선 (Qwen3 공식 가이드 준수)

### 기존 방식의 문제
모든 텍스트(문서/질문)에 동일한 임의 접두사 `"Represent this passage for retrieval: "`를
붙이고 **mean pooling**을 사용하고 있었다. 이는 Qwen3-Embedding 공식 권장과 달랐다.

### 공식 가이드 (출처)
[Qwen/Qwen3-Embedding-0.6B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

> 📌 **Tip:** We recommend that developers customize the instruct according to their
> specific scenarios, tasks, and languages. Our tests have shown that in most retrieval
> scenarios, **not using an instruct on the query side can lead to a drop in retrieval
> performance by approximately 1% to 5%.**

공식 요점:
- **질문(query)** 에만 instruct를 적용한다. 형식: `Instruct: {task}\nQuery: {query}`
- **문서(passage)** 에는 instruct를 붙이지 않고 원문 그대로 임베딩한다.
- instruct의 task 설명은 **영어**로 작성하기를 권장한다(학습이 영어 기반).
- pooling은 **last-token pooling**을 사용한다.

### 변경 내용

| 항목 | 변경 전 | 변경 후 (공식 준수) |
|------|---------|---------------------|
| 문서(passage) 임베딩 | `"Represent this passage..."` 접두사 | **접두사 없음 (원문 그대로)** |
| 질문(query) 임베딩 | 동일 접두사 | **`Instruct: ...\nQuery: ...` 형식** |
| Pooling | mean pooling | **last-token pooling** |
| L2 정규화 | 적용 | 적용 (유지) |

채택한 instruct task 문구(법률 도메인 반영):
```
Given a user's legal question, retrieve relevant legal provisions that answer it
```

### 수정 파일
- `src/scripts/data_embed_to_file_qwen.py` — 문서 임베딩: 접두사 제거 + last-token pooling
- `embedding_server/embed_server.py` — 질문 임베딩: Instruct 형식 + last-token pooling

### 정합성 주의
DB(문서)와 검색(질문)은 **pooling·정규화 방식이 동일**해야 하며, instruct 적용 여부만
공식 가이드대로 비대칭(문서=무, 질문=유)으로 둔다. 임베딩 방식이 바뀌었으므로
**기존 DB 벡터 전체를 재임베딩**해야 한다. (미국+캐나다 전체)

---

## 4. RAGAS 평가 파이프라인 재설계

### 4.1 데이터셋 생성 (`ragas_generate_dataset.py`)

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 생성 LLM | gemini-2.5-pro 고정 | 서비스와 동일한 `settings.GCP_MODEL_NAME` |
| 생성 임베딩 | text-embedding-005 | 자체 호스팅 Qwen3 |
| 샘플링 | `country_id==1`만 50개 | **국가별 층화 샘플링** (국가당 30개) |
| 질문 생성 | 전체 20개 일괄 | **국가별 7개씩 생성 + country_id 태깅** (6국 = 42개) |
| 질문 유형 | SingleHop/MultiHopAbstract/MultiHopSpecific | **SingleHop 70% + MultiHopAbstract 30%** (전문가형 Specific 제외) |
| llm_context | 한국어 유도만 | 서비스 시나리오(해외 체류자 법률) + 한국어 + JSON 규칙 |

**핵심 개선 — 국가별 생성 + country_id 태깅:**
질문을 국가별로 따로 생성하고 `country_id`를 함께 저장한다. 그래야 평가 단계에서
각 질문을 **해당 국가 법률로만 검색**하여 실제 서비스와 동일한 조건으로 평가할 수 있다.

> MultiHopSpecific을 제외한 이유: 조항 번호를 직접 지목하는 전문가형 질문으로,
> 일반 사용자(유학생/여행자) 타깃과 맞지 않기 때문.

### 4.2 평가 실행 (`ragas_evaluate_rag.py`)

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 검색 임베딩 | text-embedding-005 (768) | **Qwen3 (1024)** — DB와 차원·방식 일치 |
| 채점 임베딩 | text-embedding-005 | **Qwen3로 통일** (MTEB 범용 성능 우수, 일관성) |
| 검색 국가 | `country_id=1` 하드코딩 | **질문별 country_id로 검색** |
| 생성 LLM | 서비스 모델 | 서비스 모델 (유지) |
| 채점 LLM | gemini-2.5-pro | gemini-2.5-pro (유지, 편향 방지 위해 분리) |

> 검색 임베딩을 005로 두면 DB(Qwen3 1024차원)와 차원이 달라 검색 자체가 불가능했다.
> → 반드시 Qwen3로 교체해야 평가가 성립한다.

---

## 5. 모델 선정 근거 (참고)

임베딩 모델은 MTEB 법률 retrieval 벤치마크를 직접 측정해 선정하였다.

- `Qwen3-Embedding-0.6B`: **0.8677**
- `text-embedding-005`: 0.4464

법률 도메인에서 약 2배의 정확도 차이가 있어 Qwen3를 자체 호스팅으로 채택하였다.
(상세 인프라 결정은 `INFRA_DECISION.md` 참고)

---

## 6. 향후 실행 순서 (체크리스트)

### Phase A. 데이터 재구축 (집/GPU 환경)
- [ ] `TRUNCATE laws;` — 임베딩 방식이 바뀌었으므로 전체 비움
- [ ] Raw 데이터 확인: `US_Law_Data_Raw.jsonl`, `US_Fed_Law_Data_Raw.jsonl`, `Canada_Law_Data_Raw.jsonl`(BC 포함)
- [ ] `data_embed_to_file_qwen.py` 실행 — 새 방식(무접두사 + last-token)으로 전체 재임베딩
- [ ] `db_load_law_data.py` 실행 — 적재
- [ ] 검증: `SELECT country_id, COUNT(*) FROM laws GROUP BY country_id;` → 1~6 모두 존재(특히 6)

### Phase B. 평가 (VPC 내부/임베딩 서버 접근 가능 환경)
- [ ] `ragas_generate_dataset.py` 실행 — 국가별 42문제 생성
- [ ] `ragas_evaluate_rag.py --name qwen3_official` 실행 — 현재 시스템 점수 측정

### 주의사항
- 평가셋 생성·평가는 임베딩 서버(`EMBEDDING_URL`, 내부 IP)에 접근 가능한 환경에서 실행할 것.
- 과거 `eval_history.csv`의 점수는 **이전 모델·데이터·평가셋** 기준이므로 새 점수와 직접 비교하지 않는다.
  (변수 통제가 안 되므로) 모델 교체 근거는 MTEB 벤치마크로, RAGAS는 현재 성능 측정 용도로 사용한다.

---

## 7. 면접/발표용 요약 포인트

- **버그 발견·해결력**: 부분 문자열 매칭으로 BC가 Ontario로 오분류된 데이터 정합성 버그를 발견하고 수정.
- **공식 문서 기반 개선**: Qwen3 공식 가이드(query instruct, document 무접두사, last-token pooling)에 맞춰 임베딩을 재구성. 공식 팁상 query instruct 미적용 시 검색 성능 1~5% 손해.
- **평가 공정성 이해**: 국가별 데이터 불균형 → 층화 샘플링, 질문별 country_id 태깅으로 실제 서비스와 동일 조건 평가, 변수 통제(모델 비교 ≠ 평가셋 비교) 인지.
