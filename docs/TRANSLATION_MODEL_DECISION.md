# 번역 모델 선정 의사결정

> 작성일: 2026-06-10  
> 상태: 채택 결정, 서비스 적용 및 자체 평가 예정  
> 최종 선택: Google Cloud Translation LLM (`general/translation-llm`)

---

## 1. 배경

GLAW의 법령 데이터와 임베딩 검색 대상은 영어로 구성되어 있다.
사용자는 한국어로 질문하므로, 검색 전에 질문을 영어로 번역해야 한다.

현재는 답변 생성에도 사용하는 범용 Gemini 모델을 번역에 재사용하고 있다.
하지만 번역은 검색 품질에 직접 영향을 주며, 범용 답변 생성과 요구 역량이 다르다.
따라서 번역 품질에 특화된 모델을 별도로 선정하기로 하였다.

```text
한국어 질문
→ 영어 번역
→ Qwen3 질문 임베딩
→ 영어 법령 검색
→ 근거 기반 한국어 답변 생성
```

---

## 2. 후보 선정 과정

### 2.1 외부 번역 평가에서 Gemini 계열 확인

[Alconost의 2025~2026 번역 엔진 평가](https://alconost.com/en/blog/best-llm-for-translation-2026#by-language)는
실제 고객 프로젝트를 기반으로 5,632건의 번역 평가를 집계하였다.

해당 평가에서 Gemini는 다음 결과를 기록하였다.

| 평가 구분 | 결과 |
|-----------|------|
| 전체 엔진 평균 AQI | Gemini 77.7로 1위 |
| 영어 → 한국어 | Gemini 78.2로 1위 (`n=12`) |
| 법률 콘텐츠 | Gemini 84.6로 1위 (`n=5`) |

이 결과를 통해 Gemini 계열을 번역 모델의 우선 후보로 선정하였다.

다만 이 자료는 다음 한계가 있다.

- 세부 Gemini 모델명을 공개하지 않았다.
- 한국어 평가는 **영어 → 한국어** 방향이며, GLAW가 사용하는 **한국어 → 영어** 방향과 다르다.
- 한국어 및 법률 콘텐츠 표본 수가 작다.
- Alconost도 프로젝트별 자체 평가를 권장한다.

따라서 이 결과는 최종 성능 증명이 아니라, **Gemini 계열을 우선 검토할 외부 근거**로 사용한다.

### 2.2 Google 공식 가이드에서 번역 전용 모델 비교

[Google Cloud Translation 모델 비교 공식 문서](https://docs.cloud.google.com/translate/docs/advanced/compare-models)는
Translation LLM을 다음과 같이 설명한다.

> "Highest quality translation model"

또한 Translation LLM은 Gemini 기반으로 번역에 맞게 미세 조정된 모델이며,
Gemini 2.0 Flash보다 지연 시간이 2배 빠르다고 안내한다.

| 모델 | 공식 용도 | 모델 ID |
|------|-----------|---------|
| Translation LLM | 최고 품질 번역 | `general/translation-llm` |
| Neural Machine Translation | 가장 빠른 번역, 실시간·지연 민감 작업 | `general/nmt` |

[Google Cloud Translation 지원 언어 공식 문서](https://docs.cloud.google.com/translate/docs/languages)는
Translation LLM이 지원 목록 내 언어 간 번역을 지원한다고 설명하며,
한국어(`ko`)와 영어(`en`) 모두 정식 지원 언어에 포함한다.

---

## 3. 최종 결정

질문 번역 모델로 **Google Cloud Translation LLM**을 선택한다.

```text
모델 ID: general/translation-llm
서비스: Cloud Translation - Advanced
번역 방향: Korean (ko) → English (en)
적용 범위: 검색용 질문 번역
```

선정 이유는 다음과 같다.

1. 외부 평가에서 Gemini 계열이 한국어 번역과 법률 콘텐츠 번역에서 높은 성능을 보였다.
2. Google 공식 문서가 `general/translation-llm`을 최고 품질 번역 모델로 분류한다.
3. 범용 Gemini가 아니라 번역에 맞게 미세 조정된 Gemini 기반 모델이다.
4. GLAW 인프라가 GCP 기반이므로 기존 IAM, 서비스 계정, 과금 체계를 활용하기 쉽다.
5. 한국어와 영어를 모두 공식 지원한다.

`general/nmt`는 응답 속도가 더 중요한 상황에는 적합하지만,
현재 서비스에서는 번역 속도보다 **번역된 검색 질의가 법률 의미를 정확히 보존하는 것**을 우선하므로 선택하지 않았다.

---

## 4. 적용 원칙

- Translation LLM은 **검색용 질문 번역에만 사용**한다.
- 질문 재구성과 최종 답변 생성은 기존 생성 모델을 사용한다.
- 최종 답변 생성에는 번역된 질문이 아니라 사용자의 원본 한국어 질문을 전달한다.
- Translation LLM 호출 실패 시 부정확한 검색을 진행하지 않고 오류를 반환한다.

```text
질문 재구성: 생성 모델
질문 번역: general/translation-llm
법령 검색: Qwen3-Embedding-0.6B + pgvector
최종 답변: 생성 모델
```

---

## 5. 자체 검증 계획

외부 평가와 공식 설명만으로 프로젝트 성능을 확정할 수 없으므로,
적용 전후를 동일한 30문항 평가셋으로 비교한다.

| 확인 항목 | 판단 기준 |
|-----------|-----------|
| `ContextRecall` | 필요한 법령을 놓치지 않는지 |
| `ContextPrecision` | 관련 법령이 검색 상위에 배치되는지 |
| 평균 번역 시간 | 기존 범용 Gemini 번역 대비 지연 시간 |
| 평균 총 응답 시간 | 전체 사용자 응답시간 증가 여부 |
| 실패율 | 번역 API 오류 또는 빈 결과 발생 여부 |

최종 채택 여부는 `general/translation-llm` 적용 후 검색 지표가 개선되거나 유지되고,
응답시간과 비용이 프로젝트 허용 범위 안에 있는지를 기준으로 확인한다.

---

## 6. 참고 자료

- [Alconost - Best LLM for Translation in 2026](https://alconost.com/en/blog/best-llm-for-translation-2026#by-language)
- [Google Cloud Translation - Choose the right model for your application](https://docs.cloud.google.com/translate/docs/advanced/compare-models)
- [Google Cloud Translation - Language support](https://docs.cloud.google.com/translate/docs/languages)
- [Google Cloud Translation - Translation LLM](https://docs.cloud.google.com/translate/docs/translation-llm)
