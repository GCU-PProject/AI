# RAGAS 평가 파이프라인 설계

> 2026-06-01 작성, 2026-06-09 갱신 | 분야: 평가

## 배경

RAG 개선 작업의 효과를 정량적으로 측정하기 위해 RAGAS 기반 평가 파이프라인을 구축했다. 핵심 설계 원칙은 **"실제 서비스와 동일한 조건으로 평가한다"**.

## 데이터셋 설계

- 130만 조항에서 무작위 샘플링하면 서비스와 무관한 조항(농산물 등급 등)으로 무의미한 문항이 생성된다 → **사람이 검수한 seed 조항**(여행자/유학생이 실제로 물을 법한 주제)만 사용
- 실제 서비스는 주 선택 시 "주법 + 연방법"을 함께 검색 → 평가셋도 주 단위로 생성하고 동일한 검색 범위 적용 (CA/NY/Ontario/BC)
- seed 법령 1개당 질문 1개 생성, 환각 방지를 위해 supporting_quote가 원문에 실제 존재하는지 검증

## 지표 선정 (RAGAS 공식 문서 기반)

RAGAS 공식 문서는 특정 지표 조합을 강제하지 않는 대신 선택 원칙을 제시한다.

- [RAGAS Metrics Overview](https://docs.ragas.io/en/v0.4.2/concepts/metrics/overview/): 전체 사용자 경험을 반영하는 지표를 우선 보라.
  > "Focus first on metrics reflecting overall user satisfaction."
- 같은 문서: 약한 신호를 주는 지표를 늘리면 의사결정이 흐려진다.
  > "Avoid a proliferation of metrics that provide weak signals and impede clear decision-making."
- [RAGAS 공식 RAG 평가 예제](https://docs.ragas.io/en/stable/getstarted/rag_eval/)는 단순 RAG 평가에 다음 3개 지표를 사용한다:
  ```python
  from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
  ```

이 원칙에 따라 4개로 압축했다. 각 지표의 공식 정의:

| 지표 | 대상 | 역할 | 공식 문서 원문 |
|------|------|------|--------------|
| [FactualCorrectness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/factual_correctness/) | Baseline + RAG 공통 | RAG 도입 전후 답변 품질 비교 (유일한 공통 지표) | "compares and evaluates the factual accuracy of the generated `response` with the `reference`" |
| [ContextRecall](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/context_recall/) | RAG만 | 필요한 법령을 놓치지 않았는가 | "recall is about not missing anything important" |
| ContextPrecision | RAG만 | 관련 법령이 검색 상위에 배치되는가 | (공식 예제 ContextPrecision 계열 대응) |
| [Faithfulness](https://docs.ragas.io/en/v0.4.2/concepts/metrics/available_metrics/faithfulness/) | RAG만 | 답변이 검색 근거에 충실한가 (환각 점검) | "A response is considered faithful if all its claims can be supported by the retrieved context" |

제외한 지표와 이유:

- `AnswerRelevancy`: 질문-답변 관련성을 보지만, 법률 서비스에서는 "관련 있어 보임"보다 **사실적으로 맞는지**와 **근거에 충실한지**가 더 강한 신호다.
- `AnswerCorrectness`: 목적이 유사하나 공식 RAG 예제가 사용하는 `FactualCorrectness`로 대체했다.

Baseline(RAG 미적용)은 검색이 없으므로 검색 지표를 측정하지 않고, RAG 전후 공통 비교에는 FactualCorrectness만 사용한다.

## 채점 모델 고정

채점 LLM은 전 실험에서 `gemini-3-flash-preview` **하나로 고정**했다.

1. **실험 간 비교 가능성 (가장 중요)**: 채점자가 바뀌면 점수 변화가 "시스템 개선" 때문인지 "채점자 변경" 때문인지 구분할 수 없다. 절대 점수가 아닌 **실험 간 상대 비교**가 목적이다.
2. **공식 문서 근거**: RAGAS는 평가 모델 선택에 대해 다음 한 문장만 명시하며, "더 강한 모델을 쓰라"는 권장은 없다.
   > "You may choose any model as evaluator LLM for evaluation."
3. **비용 효율**: 다수의 반복 실험이 예정되어 있어 고비용 모델 채점은 부담이 크다.

## 시행착오

**1. 채점 모델의 thinking 토큰 폭발**
FactualCorrectness 1회 호출에 25만 토큰($0.76)이 소비됐다. 원인은 채점 모델의 thinking 기본값(high) — 2문장을 claim 2개로 나누는 단순 작업에 6만 토큰의 추론을 수행했다. `thinking_budget=0`으로 완전 비활성화하여 문항당 ~5천 토큰으로 정상화했다.

**2. 불필요한 컬럼이 채점 비용 유발**
eval_dataset에 reference_contexts(법령 원문 전체)를 포함하자 FactualCorrectness가 이를 claim 분해 대상으로 처리해 토큰이 폭발했다. 지표가 실제로 사용하는 컬럼만 전달해야 한다.

**3. 합성 reference의 편향**
seed 조항 기반으로 생성된 정답이 짧고 간결해서(44~288자), 간결하게 답하는 모델이 FactualCorrectness에서 유리해지는 편향이 있다. 절대 점수가 아닌 상대 비교로만 해석하고, Faithfulness를 함께 본다.

## 배운 점

- 평가 조건(채점자, 데이터셋, 검색 범위)을 하나라도 바꾸면 이전 결과와 비교 불가. **변수 통제가 평가의 전부다.**
- LLM 채점 비용은 지표 수 × 문항 수 × thinking 설정에 민감하다. 채점용 LLM은 추론을 꺼야 한다.
- 합성 데이터셋은 빠르지만 reference 품질 한계를 인지하고 해석해야 한다.
