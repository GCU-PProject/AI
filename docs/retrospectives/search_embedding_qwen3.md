# 임베딩 방식 개선 — Qwen3 공식 가이드 준수

> 2026-06-01 | 분야: 검색 (임베딩)

## 배경

임베딩 모델은 MTEB 법률 retrieval 벤치마크를 직접 측정해 선정했다.

| 모델 | 법률 retrieval 점수 |
|------|------|
| **Qwen3-Embedding-0.6B** (자체 호스팅) | **0.8677** |
| text-embedding-005 (GCP) | 0.4464 |

약 2배 차이로 Qwen3를 채택했는데, 사용 방식이 공식 가이드와 어긋나 있었다.

## 문제

모든 텍스트(문서/질문)에 동일한 임의 접두사 `"Represent this passage for retrieval: "`를 붙이고 mean pooling을 사용하고 있었다.
[Qwen3-Embedding 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)의 권장 방식은 다르다.

> 📌 **공식 문서 원문:** "We recommend that developers customize the instruct according to their specific scenarios, tasks, and languages. Our tests have shown that in most retrieval scenarios, **not using an instruct on the query side can lead to a drop in retrieval performance by approximately 1% to 5%.**"

공식 가이드 요점:

- **질문(query)에만** instruct를 적용한다. 형식: `Instruct: {task}\nQuery: {query}`
- **문서(passage)** 에는 instruct를 붙이지 않고 원문 그대로 임베딩한다
- instruct의 task 설명은 **영어**로 작성 (모델 학습이 영어 기반)
- pooling은 **last-token pooling**을 사용한다

## 변경

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 문서 임베딩 | 임의 접두사 | 접두사 없음 (원문) |
| 질문 임베딩 | 동일 접두사 | Instruct 형식 (법률 도메인 task 문구) |
| Pooling | mean | last-token |

채택한 instruct (법률 도메인 반영):

```
Given a user's legal question, retrieve relevant legal provisions that answer it
```

정합성 주의: DB(문서)와 검색(질문)은 **pooling·정규화 방식이 동일**해야 하며, instruct 적용 여부만 공식 가이드대로 비대칭(문서=무, 질문=유)으로 둔다. 임베딩 방식이 바뀌었으므로 **DB 전체(미국+캐나다)를 재임베딩**했다.

## 배운 점

- 모델은 공식 권장 사용법까지 따라야 벤치마크 성능이 나온다. "돌아간다"와 "제대로 쓴다"는 다르다.
- 문서와 질문의 임베딩 정합성(pooling·정규화 동일)이 깨지면 검색 자체가 무의미해진다.
