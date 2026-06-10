# 회고 모음

> 프로젝트 진행 중 발견한 문제와 의사결정 기록.
> 파일명 규칙: `<분야>_<주제>.md`

| 분야 | 문서 | 한 줄 요약 |
|------|------|-----------|
| 데이터 | [data_bc_loading_bug.md](data_bc_loading_bug.md) | 부분 문자열 매칭 버그로 BC 법률이 Ontario로 오분류 |
| 검색 | [search_embedding_qwen3.md](search_embedding_qwen3.md) | Qwen3 공식 가이드 준수로 임베딩 방식 재구성 |
| 검색 | [search_vector_limit_hybrid.md](search_vector_limit_hybrid.md) | 막연한 질문에서 벡터 검색 단독의 랭킹 한계 발견 |
| 검색 | [search_db_vector_index.md](search_db_vector_index.md) | 검색 5초의 원인 — 벡터 인덱스 누락 발견과 HNSW 적용 |
| 평가 | [eval_ragas_design.md](eval_ragas_design.md) | RAGAS 지표 선정과 평가 파이프라인 설계 근거 |
| 모델 | [model_translation_selection.md](model_translation_selection.md) | 번역 모델 분리 — Cloud Translation LLM 채택 |
| 모델 | [model_thinking_tradeoff.md](model_thinking_tradeoff.md) | RAGAS 비교 실험 전 과정과 생성 모델 선정 (추론 트레이드오프) |
