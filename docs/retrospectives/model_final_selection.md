# 최종 생성 모델 확정 — gemini-3.1-flash-lite vs 3.5-flash

> 2026-06-11 | 분야: 모델 | HNSW 조건에서 최종 후보 2개 비교

## 배경

`model_thinking_tradeoff.md`에서 gemini-3.1-flash-lite가 1차 선택됐으나, 당시 측정은 풀스캔 벡터 검색 기준이었다. HNSW 인덱스 도입 + ef_search=100 설정 후 동일 조건에서 최종 후보 두 개를 재비교했다.

비교 조건 (모두 고정):
- 번역: Cloud Translation LLM
- 검색: HNSW, ef_search=100, top_k=5, L2 threshold=0.90
- 생성: temperature=0, max_tokens=4096

## 비교 결과

| 실험 | CR | CP | Faithfulness | FactualCorrectness | 총응답시간 |
|------|----|----|-------------|-------------------|----------|
| 3.5-flash_low_tr_hnsw100 | 0.8917 | 0.8228 | 0.7311 | **0.5490** | 9.03s |
| **3.1-flash-lite_tr_hnsw100** | 0.8917 | 0.8228 | **0.7502** | 0.5150 | **4.70s** |

- CR/CP 동일 — 번역·검색 조건이 같으면 검색 결과가 같으므로 예상된 결과
- Faithfulness: 3.1-lite 우위 (+0.019)
- FactualCorrectness: 3.5-flash 우위 (+0.034)
- **총응답시간: 3.1-lite가 약 2배 빠름 (4.70s vs 9.03s)**

## 결정 — gemini-3.1-flash-lite 확정

품질 차이는 0.02~0.03 수준으로 30문항 기준 노이즈 범위다. 반면 응답시간 차이(4.3초)는 사용자가 직접 체감하는 차이다.

법률 서비스 특성상 Faithfulness(검색 근거 충실도)를 FC보다 우선하는 기준(`model_thinking_tradeoff.md` 참고)도 3.1-lite에 유리하다. **품질 동등 + Faithfulness 우위 + 2배 빠른 속도** → 3.1-flash-lite 확정.

## 배운 점

- HNSW 도입 후 검색 시간이 5.2s → 1.5s로 줄면서 응답시간 비교의 의미가 달라졌다. 검색 비중이 줄자 생성 속도 차이(3.1-lite가 빠른 경량 모델)가 더 부각됐다.
- 인프라 변경(인덱스, ef_search) 후에는 기존 측정값을 재신뢰하지 말고 동일 조건에서 재측정해야 한다.
