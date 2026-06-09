# RAG 성능 평가 가이드

## 평가 목적

RAG 검색 설정과 생성 모델을 변경하며 검색 품질, 답변 품질, 응답시간을 비교한다.
한 번의 실험에서는 하나의 조건만 변경한다.

## 고정 조건

모든 실험에서 다음 조건을 동일하게 유지한다.

- 평가 데이터셋: `data/ragas_testset.csv`
- 채점 모델: `gemini-3-flash-preview`
- 채점 모델 추론: `thinking_budget=0`
- 임베딩 모델: `Qwen3-Embedding-0.6B`
- 검색 범위: 선택한 주법 + 해당 국가 연방법
- 생성 파라미터: `temperature=0`, `max_tokens=4096`, `top_k=20`, `top_p=0.7`

채점 모델은 생성 모델보다 작더라도 실험 간 비교를 위해 고정한다.

## 조정할 항목

다음 순서로 하나씩 변경하며 평가한다.

1. 검색 문서 수: `RAG_TOP_K` (`3`, `5`, `7`)
2. 검색 거리 임계값: `RAG_MAX_DISTANCE_THRESHOLD` (`0.80`, `0.85`, `0.90`)
3. 생성 모델: `.env`의 `GCP_MODEL_NAME`
4. 서비스 답변 생성 추론 수준: `minimal`, `low`, `medium`
5. 답변 프롬프트: 길이, 근거 사용 강도, 추측 방지 규칙, 답변 형식

생성용 `top_k`, `top_p`는 검색 문서 수를 의미하는 `RAG_TOP_K`와 다르다.
법률 답변의 일관성과 실험 해석을 위해 생성 파라미터는 우선 고정한다.

추후에는 L2 거리, 코사인 유사도, 하이브리드 검색, reranker 등 검색 알고리즘을 비교한다.

## 평가 지표

모든 RAG 실험에서 다음 지표를 기록한다.

| 지표 | 확인 내용 |
|---|---|
| ContextRecall | 필요한 법령을 빠뜨리지 않고 검색했는가 |
| ContextPrecision | 검색된 법령 중 관련 법령의 비율과 순위가 적절한가 |
| Faithfulness | 답변이 검색된 법령에 근거하는가 |
| FactualCorrectness | 생성 답변이 기준 정답과 사실적으로 일치하는가 |

RAG 미적용 Baseline은 검색 문서가 없으므로 `FactualCorrectness`만 측정한다.

## 응답시간 측정

RAGAS 채점시간은 사용자 응답시간에 포함하지 않는다.

```text
총 응답시간 = 질문 번역 + 임베딩 및 검색 + 답변 생성
```

각 실험에서 다음 값을 기록한다.

- 평균 응답시간
- P50 응답시간
- P95 응답시간
- 최소/최대 응답시간
- 성공/실패 문항 수

개별 요청 분석은 LangSmith를 사용하고, 실험 간 비교 결과는
`data/eval_results/latency_history.csv`에 기록한다.
시간 통계는 성공한 요청만으로 계산하며, 전체 문항 수와 실패 문항 수는 별도로 기록한다.

응답시간은 `ragas_evaluate_rag.py`와 `ragas_evaluate_baseline.py` 실행 시
답변 생성 단계에서 자동 측정된다. RAGAS 채점에 들어가기 전에 저장하므로
채점 실패 시에도 응답시간 기록은 유지된다.

- RAG 평가: 번역, 검색, 답변 생성, 전체 시간을 각각 기록
- Baseline 평가: 답변 생성과 전체 시간만 기록하며 번역·검색 시간은 `0`
- FastAPI HTTP 요청 시간과 RAGAS 채점 시간은 측정값에 포함하지 않음

## 실행 방법

```bash
# 문항 1개로 동작 확인
uv run python src/scripts/ragas_evaluate_rag.py --name smoke_rag --limit 1

# 문항 5개로 후보 설정 비교
uv run python src/scripts/ragas_evaluate_rag.py --name rag_k5_d090 --limit 5

# 전체 데이터셋 최종 평가
uv run python src/scripts/ragas_evaluate_rag.py --name final_rag_k5_d090
```

## 추천 실험 순서

```text
TOP_K 3/5/7 비교
→ 거리 임계값 0.80/0.85/0.90 비교
→ 생성 모델 비교
→ 서비스 추론 수준 비교
→ 프롬프트 개선
→ 최종 설정 전체 평가
```

실험명에는 변경한 조건을 기록한다.

```text
rag_k5_d090
rag_k3_d090
rag_k5_d085
final_gemini35_k5_d085
```
