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
| 생성 LLM | gemini-2.5-pro 고정 | **`gemini-3.5-flash` 고정** |
| 생성 임베딩 | text-embedding-005 | 자체 호스팅 Qwen3 |
| 샘플링 | `country_id==1`만 50개 | **사람이 검수한 seed 문서만 사용** (주법 + 연방법 포함) |
| 질문 생성 | 전체 20개 일괄 | **seed 법령 1개당 1문항 생성** (4개 주 × 5개 = 총 20문항) |
| 질문 유형 | SingleHop/MultiHopAbstract/MultiHopSpecific | **단일 법령 기반 질문** |
| llm_context | 한국어 유도만 | 서비스 시나리오(해외 체류자 법률) + 한국어 + JSON 규칙 |

**핵심 개선 — 주별 생성 + 주 country_id 태깅:**
실제 서비스는 사용자가 주/지역을 선택하면 해당 주법과 같은 국가의 연방법을 함께 검색한다.
따라서 평가셋도 연방 단독(1, 4)을 만들지 않고, 주(2, 3, 5, 6)만 대상으로 생성한다.
각 주의 질문 생성 재료에는 사람이 검수한 seed 문서만 포함한다.
생성 단계에서는 seed 법령 하나씩 `testset_size=1`로 처리해 **법령 1개당 질문 1개**가 나오도록 하고,
생성된 질문은 **주 country_id**로 태깅한다.

| 평가 대상 주 | 질문 생성 재료 | 저장되는 country_id | 평가 시 검색 범위 |
|--------------|----------------|---------------------|-------------------|
| 캘리포니아 | 2 + 1(미국 연방) | 2 | 2 + 1 |
| 뉴욕 | 3 + 1(미국 연방) | 3 | 3 + 1 |
| 온타리오 | 5 + 4(캐나다 연방) | 5 | 5 + 4 |
| BC | 6 + 4(캐나다 연방) | 6 | 6 + 4 |

> 여러 법령을 엮는 질문을 제외한 이유: seed 문서들은 주제별 단일 조항 질문에 적합하고
> 조항 간 의미적 연결은 약하다. 또한 일반 사용자(유학생/여행자)는 여러 조항을 복합 추론하는 질문보다
> 단일 조항 근거로 답할 수 있는 실생활 질문을 주로 하므로, 단일 법령 기반 질문이 현재 평가 목적에 부합한다.

### 4.2 평가 실행 (`ragas_evaluate_rag.py`)

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 검색 임베딩 | text-embedding-005 (768) | **Qwen3 (1024)** — DB와 차원·방식 일치 |
| 채점 임베딩 | text-embedding-005 | **Qwen3로 통일** (MTEB 범용 성능 우수, 일관성) |
| 검색 국가 | `country_id=1` 하드코딩 | **질문별 country_id로 검색** |
| 생성 LLM | 서비스 모델 | **`gemini-3.5-flash` 고정** |
| 채점 LLM | gemini-2.5-pro | **`gemini-3.5-flash` 고정** (생성과 동일, 아래 근거 참고) |
| 평가 지표 | AnswerCorrectness / AnswerRelevancy / ContextPrecision / ContextRecall | **FactualCorrectness(공통 비교) + ContextRecall/ContextPrecision/Faithfulness(RAG 내부 진단)** |

> 검색 임베딩을 005로 두면 DB(Qwen3 1024차원)와 차원이 달라 검색 자체가 불가능했다.
> → 반드시 Qwen3로 교체해야 평가가 성립한다.

### 4.3 평가 지표 선정 근거 (공식 문서 기반)

RAGAS 공식 문서는 특정 지표 조합을 강제하지 않는다. 대신 지표 선택 원칙을 제시한다.

- [RAGAS Metrics Overview](https://docs.ragas.io/en/v0.4.2/concepts/metrics/overview/)는
  우선적으로 전체 사용자 경험을 반영하는 end-to-end 지표를 보라고 설명한다.
  - 원문: **"Focus first on metrics reflecting overall user satisfaction."**
- 같은 문서는 약한 신호를 주는 지표를 많이 늘리면 의사결정이 흐려질 수 있으므로,
  적은 수의 강한 지표를 선택하라고 설명한다.
  - 원문: **"Avoid a proliferation of metrics that provide weak signals and impede clear decision-making."**
- [RAGAS RAG 평가 공식 예제](https://docs.ragas.io/en/stable/getstarted/rag_eval/)는
  단순 RAG 시스템 평가 예시에서 다음 3개 지표를 사용한다. 또한 평가 모델은
  사용자가 선택할 수 있다고 설명한다.
  - 원문: **"You may choose any model as evaluator LLM for evaluation."**

```python
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
```

공식 문서의 각 지표 설명을 기준으로 보면 다음과 같다.

- [Context Recall](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/context_recall/)은
  관련 문서 또는 정보가 얼마나 성공적으로 검색되었는지 측정하며,
  중요한 결과를 놓치지 않는 데 초점을 둔다.
  - 원문: **"In short, recall is about not missing anything important."**
- [Faithfulness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/faithfulness/)는
  생성 답변이 검색된 컨텍스트와 사실적으로 일관적인지 측정한다.
  공식 문서는 답변의 모든 주장이 검색 컨텍스트로 뒷받침될 수 있으면 faithful하다고 설명한다.
  - 원문: **"A response is considered faithful if all its claims can be supported by the retrieved context."**
- [Factual Correctness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/factual_correctness/)는
  생성 답변의 사실 정확도를 기준 정답(reference)과 비교해 평가한다.
  응답과 기준 정답을 claim 단위로 나눈 뒤 사실적 겹침을 판단한다.
  - 원문: **"`FactualCorrectness` is a metric that compares and evaluates the factual accuracy of the generated `response` with the `reference`."**

수동 평가셋은 자동 생성 결과를 보완하기 위한 별도 실험셋으로 둔다.
단일 주제(예: 음주운전)만으로 구성하면 특정 도메인 검색 성능을 보기는 좋지만,
서비스 전체 품질을 대표한다고 말하기는 어렵다. 발표용 최종 평가셋은 일반 사용자가 실제로 물을 법한
여러 주제(교통/체류·비자/벌금·처벌/생활 법률 등)를 섞고, 질문도 지나치게 전문적이지 않게 구성한다.
따라서 본 프로젝트에서는 지표를 **공통 비교 지표**와 **RAG 내부 진단 지표**로 분리한다.
Baseline은 검색을 하지 않으므로 `ContextRecall`, `Faithfulness`를 계산하지 않고,
RAG 적용 전후 답변 품질 비교에는 `FactualCorrectness`만 사용한다.

| 선택 지표 | 적용 대상 | 공식 예제 대응 | 선택 이유 |
|-----------|-----------|----------------|-----------|
| `FactualCorrectness` | Baseline + RAG 공통 | `FactualCorrectness` | RAG 미적용 답변과 RAG 적용 답변을 같은 기준 정답(reference)에 대해 비교할 수 있다. RAG 도입 전후 답변 품질 변화량을 보는 지표다. |
| `ContextRecall` | RAG만 | `LLMContextRecall` 계열 | 필요한 법령 근거를 검색했는지 확인한다. 검색 실패는 이후 답변 품질을 직접 떨어뜨린다. |
| `ContextPrecision` | RAG만 | `ContextPrecision` 계열 | 검색된 법령 중 실제 질문에 필요한 문서가 앞쪽에 잘 배치되는지 확인한다. 불필요한 문서가 많으면 답변 생성 품질과 비용이 함께 나빠질 수 있다. |
| `Faithfulness` | RAG만 | `Faithfulness` | 답변이 검색된 법령 근거에 충실한지 확인한다. 법률 서비스에서 가장 위험한 환각을 점검하는 핵심 지표다. |

`FactualCorrectness`는 RAGAS가 생성한 합성 기준 답변과 비교하므로 한계가 있다.
서비스 답변이 더 짧거나 표현 방식이 다르면 절대 점수가 낮게 나올 수 있다.
따라서 이 지표는 "정답률 그 자체"로 해석하지 않고, 동일한 질문·동일한 기준 답변에서
**Baseline 대비 RAG가 얼마나 개선되었는지 보는 상대 비교 지표**로 사용한다.

제외한 지표:

- `AnswerRelevancy`: 질문과 답변의 관련성을 보지만, 법률 서비스에서는 "관련 있어 보임"보다
  **사실적으로 맞는지**와 **검색 근거에 충실한지**가 더 강한 신호다.
- `AnswerCorrectness`: 목적이 유사하지만 공식 RAG 예제에서 사용하는 `FactualCorrectness`로 대체했다.

> 결론: 공식 문서의 원칙처럼 지표를 많이 늘리기보다,
> **Baseline 대비 답변 품질 향상은 FactualCorrectness**로 보고,
> **RAG 내부 품질은 ContextRecall/ContextPrecision/Faithfulness**로 진단한다.

### 4.4 채점 모델 선정 근거 (생성·채점 단일 모델)

채점 LLM을 생성과 동일한 **`gemini-3.5-flash`**로 고정하였다. 근거는 다음과 같다.

1. **반복 실험 간 비교 가능성 (가장 중요)**
   - 평가는 RAG 개선 전후의 성능 변화를 측정하는 것이 목적이다.
   - 채점 모델이 실험마다 바뀌면 점수 변화가 "RAG 개선" 때문인지 "채점자 변경" 때문인지
     구분할 수 없다. → 채점 모델은 처음부터 끝까지 **하나로 고정**해야 한다.
   - 절대 점수보다 **개선 전후의 상대적 변화량**을 측정하는 데 목적을 둔다.

2. **RAGAS 공식 가이드 — 단일 모델 사용 (인용)**
   - [RAGAS 공식 RAG 평가 예제](https://docs.ragas.io/en/stable/getstarted/rag_eval/)는
     **생성과 채점에 동일한 모델**을 사용한다.

     ```python
     # 생성용 LLM
     llm = ChatOpenAI(model="gpt-4o")
     # 평가용 LLM — 같은 llm 객체를 그대로 래핑
     evaluator_llm = LangchainLLMWrapper(llm)
     ```

   - RAGAS 공식 문서는 평가 모델 선택에 대해 다음 한 문장만 명시한다.
     > "You may choose any model as evaluator LLM for evaluation."
   - 즉 "더 강한 모델을 쓰라"는 공식 권장은 없으며, 모델 선택은 전적으로 사용자 재량이다.
     (`customize_models` 페이지에도 모델 선택 기준에 대한 권장은 없고 설정 방법만 안내)

3. **비용 효율**
   - 다수의 개선 실험을 반복할 예정이므로, 고비용 모델(예: Pro 계열)로 매번 채점하면 비용 부담이 크다.
   - 발표까지 남은 기간이 짧으므로 모델 변경 변수를 줄이고, Flash 단일 모델로 빠르게 반복 평가한다.

> ⚠️ 한계: 생성·채점이 같은 모델이면 자기 출력을 후하게 평가하는 self-evaluation bias가 있을 수 있다.
> 다만 본 평가의 목적은 절대 점수가 아닌 **A/B 상대 비교**이므로, 양쪽에 동일하게 작용하는 편향은 비교 결과에 영향을 주지 않는다.

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
- [x] `ragas_testset_manual.csv` 음주운전 검증셋 구성 — 4개 주 × 1문항 = 총 4문항
- [ ] 수동 평가셋 확장 — 여러 주제와 일반 사용자형 질문을 각 주별로 추가
- [ ] `ragas_evaluate_baseline.py --name baseline_no_rag` 실행 — RAG 미적용 기준점 측정 (`FactualCorrectness`)
- [ ] `ragas_evaluate_rag.py --name qwen3_official` 실행 — RAG 적용 점수 측정 (`FactualCorrectness` + `ContextRecall` + `ContextPrecision` + `Faithfulness`)

### 주의사항
- 평가셋 생성·평가는 임베딩 서버(`EMBEDDING_URL`, 내부 IP)에 접근 가능한 환경에서 실행할 것.
- 과거 `eval_history.csv`의 점수는 **이전 모델·데이터·평가셋** 기준이므로 새 점수와 직접 비교하지 않는다.
  (변수 통제가 안 되므로) 모델 교체 근거는 MTEB 벤치마크로, RAGAS는 현재 성능 측정 용도로 사용한다.

---

## 7. 면접/발표용 요약 포인트

- **버그 발견·해결력**: 부분 문자열 매칭으로 BC가 Ontario로 오분류된 데이터 정합성 버그를 발견하고 수정.
- **공식 문서 기반 개선**: Qwen3 공식 가이드(query instruct, document 무접두사, last-token pooling)에 맞춰 임베딩을 재구성. 공식 팁상 query instruct 미적용 시 검색 성능 1~5% 손해.
- **평가 공정성 이해**: 실제 서비스와 동일하게 주 선택 시 주법+연방법을 함께 검색하도록 평가셋 생성·평가 조건을 맞춤. Baseline 대비 개선은 `FactualCorrectness`, RAG 내부 품질은 `ContextRecall`/`ContextPrecision`/`Faithfulness`로 분리해 해석.
