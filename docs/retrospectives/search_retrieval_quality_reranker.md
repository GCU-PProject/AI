# 검색 품질 진단 및 리랭커 도입 검토

> 2026-06-12 | 분야: 검색 | rewrite 검증, casual testset eval, threshold 조정, 리랭커 검토

## rewrite eval 결과 (casual testset)

| 실험 | CR | CP | Faithfulness | FC |
|------|----|----|-------------|-----|
| casual baseline | 0.8056 | 0.7162 | 0.7169 | 0.3700 |
| **casual rewrite** | 0.7889 | **0.8253** | **0.7616** | 0.3320 |

- Faithfulness +0.0447, CP +0.1091 — 핵심 지표 개선
- CR -0.0167 — 미미한 하락, 허용 범위
- rewrite가 막연한 실사용자 질문에서 실질적으로 도움이 된다는 것을 확인

## casual testset 생성 배경

기존 ragas_testset_2.csv는 법률 문서에서 자동 생성되어 질문이 지나치게 공식적이었다. 실제 사용자(여행자·유학생)의 막연한 표현을 반영한 30개 질문셋을 수동으로 제작했다. 이 테스트셋이 실 서비스 품질을 더 정확하게 반영한다.

## 수동 품질 테스트 (manual_test.py) 결과

18개 질문, 서버: https://api.glaw.site/ai

- 검색성공: 16/18
- 막연: 7/8, 정상: 6/6, 엣지: 3/4

검색은 됐지만 답변 품질이 낮은 케이스 식별:

| 케이스 | 현상 |
|--------|------|
| [CA] 대마초 | 방조죄 조항만 검색됨, Cannabis 합법화 조항 누락 |
| [NY] 총기 소지 | 사냥 규정만 검색됨, 형사법 총기 소지 조항 누락 |
| [NY] 대마초 | Battery Park City 공원 규정만 검색됨 |
| [NY] 직장 해고 | 검색 결과 0건 |

## 원인 진단 (debug_retrieval.py)

각 케이스별 번역·rewrite 출력과 threshold 필터 전 상위 10개 결과를 출력하는 진단 스크립트를 작성해 원인을 분석했다.

**결론: rewrite는 정상 동작. threshold 0.90이 핵심 원인.**

관련 조항들이 실제로 존재하지만 L2 거리 0.91~0.93 구간에 있어 전부 필터링되고 있었다.

```
[CA] 대마초 Cannabis ARTICLE 2: 0.9108  (0.0108 차이로 탈락)
[NY] 총기 265.01-B Criminal Possession: 0.9117
[NY] 대마초 222.15 Personal Cultivation: 0.9250
[NY] 직장 해고 Disqualification for Benefits: 0.9189
```

## 리랭커 검토 및 결론

리랭커(cross-encoder) 도입을 검토했다. cross-encoder는 (query + 문서)를 함께 입력해 관련도를 재채점하므로 bi-encoder 기반 벡터 검색보다 순위 품질이 높다. 단, k를 20~30으로 늘려야 효과가 있고 레이턴시가 1~2초 추가된다.

**내일 배포 기준으로 리랭커는 보류.** threshold 조정으로 대부분의 케이스를 커버할 수 있고, 하루 안에 새 인프라를 붙여 검증하기엔 리스크가 크다.

## threshold 0.95 적용 결정

| 파라미터 | 변경 전 | 변경 후 |
|---------|--------|--------|
| RAG_MAX_DISTANCE_THRESHOLD | 0.90 | **0.95** |
| RAG_TOP_K | 7 | 7 (유지) |

근거:
- k=7로 가져온 상위 7개 안에 관련 조항이 있는데 threshold에서 필터링되는 경우가 대부분
- threshold만 올리면 k=7 상한에 막혀 추가 노이즈 유입이 제한됨
- 프롬프트 가드레일("근거 자료만 사용, 관련 없으면 답변 불가")이 노이즈 문서를 LLM 수준에서 걸러냄

## 내일 계획

threshold 0.95 배포 후 eval 수치 확인. 품질 개선이 불충분하면:

1. **리랭커 도입** (Cohere Rerank API 또는 BGE-Reranker 임베딩 서버 추가)
2. **k=7 → 20** 으로 확장 (리랭커의 재정렬 효과를 극대화하려면 후보 수 확보 필요)
3. casual testset으로 재eval

## 배운 점

- rewrite가 올바른 키워드를 생성하더라도 threshold가 관련 조항을 잘라낼 수 있다. threshold와 k는 함께 설계해야 한다.
- 자동 생성 testset은 문서 언어를 반영해 실 사용자 질문과 다를 수 있다. CR이 실제보다 높게 나오는 bias가 있다. 수동 casual testset이 실서비스 품질의 더 신뢰할 수 있는 지표다.
- 리랭커는 벡터 검색의 순위 문제를 해결하지만, k를 늘리지 않으면 후보 pool이 같아서 효과가 제한된다.
