# 인프라 구성 회고 — 임베딩 모델 & 서버 통합

> 주제: 임베딩 모델 처리 방식 + 비용 최소화를 위한 서버 통합

---

## 1. 문제 발견 — 임베딩 차원 불일치

| 항목 | 모델 | 차원 |
|---|---|---|
| DB에 저장된 데이터 임베딩 | Qwen3-Embedding-0.6B | 1024 |
| 서비스 코드의 쿼리 임베딩 | text-embedding-005 (API) | 768 |

두 모델은 차원뿐 아니라 **벡터 공간 자체가 달라** 혼용 시 유사도 계산이 무의미. 0 패딩으로 차원만 맞추는 것도 불가.

## 2. 모델 선택 — Qwen3 자체 호스팅 채택

MTEB 법률 retrieval 벤치마크(`mteb_legal_retrieval_leaderboard.csv`) 기준:
- **Qwen3-Embedding-0.6B: 0.8677** (233개 중 상위권)
- text-embedding-005: 0.4464 (174위, 하위권)

→ 법률 도메인 정확도가 약 2배. 데이터를 005로 재임베딩(66만 건, API 제한으로 수일)하는 대신, **Qwen3를 서버에 직접 설치(자체 호스팅)** 하기로 결정. 실무 법률 AI(Harvey AI 등)도 정확도·보안 때문에 자체 임베딩 인프라를 운영함.

## 3. 최종 서버 구성 (분리 유지 + 임베딩 우측사이징)

> 통합(5→3대)은 e2-small 2대(~$34/월)만 절감되어 마이그레이션 수고 대비 실익이 적다고 판단.
> **기존 분리 구성을 유지**하되, 가장 큰 낭비였던 **임베딩 VM을 e2-standard-4 → e2-standard-2로 우측사이징**.
> 이 변경 하나가 통합보다 더 큰 절감(~$63/월)이며, 작업은 머신 유형 변경 한 번뿐.

| 서버 | 머신 타입 | 디스크 | 외부 IP | 태그 | 수신 허용 (방화벽) |
|---|---|---|---|---|---|
| Nginx | e2-micro | 10GB | **O** (퍼블릭) | `web` | 인터넷 → 80, 443 |
| Spring | e2-small | 기본 | X | `spring` | web → 8080 |
| FastAPI | e2-small | 기본 | X | `fastapi` | web → 8000 |
| 임베딩 | **e2-standard-2** (8GB) | **30GB** | X | `embedding` | **fastapi → 8081** |
| DB | e2-standard-2 (8GB) | 100GB | X | `db` | (spring·fastapi) → 5432 |

```
인터넷 ─80/443→ [Nginx] ─→ [Spring]:8080 ─┐
                    └────→ [FastAPI]:8000 ─┼─→ [DB]:5432
                                  │
                                  └─→ [임베딩]:8081 (Qwen3, 도커)
```

### 근거 요약
- **임베딩 VM은 e2-standard-2면 충분**: Qwen3-Embedding-0.6B 단독 메모리 ~5GB(FP32 모델 2.4 + PyTorch 1.5 + OS 1). 8GB면 여유. e2-standard-4(16GB)는 단독 운영 시 절반도 못 써 낭비.
- **Spring/FastAPI는 e2-small**: Spring(JVM)은 기본 1GB+라 e2-micro(1GB)면 OOM 위험. FastAPI도 e2-micro는 공유코어라 응답 불안정. → 둘 다 small.
- **Nginx만 e2-micro**: 리버스 프록시는 RAM 50MB도 안 씀.
- **디스크**: 임베딩 30GB(PyTorch+모델), DB는 기존 100GB 유지(GCP 디스크 축소 불가, 차액 미미).

### 비용
```
Nginx(e2-micro ~$8) + Spring(e2-small $17) + FastAPI(e2-small $17)
+ 임베딩(e2-standard-2 ~$64) + DB(e2-standard-2+100GB $75)
≈ $181/월 (기존 임베딩 e2-standard-4 기준 $253 대비 ~$72 절감)
+ 미사용 시 VM stop → 추가 대폭 절감 (데모 환경 핵심 절감 수단)
```

## 4. 보안 (최소 설계)
- Nginx만 퍼블릭 서브넷 + 외부 IP. 나머지 4대는 외부 IP 없음 → 인터넷에서 도달 경로 자체가 없음.
- 방화벽 사슬: 인터넷 → (80/443) Nginx → (8080/8000) Spring·FastAPI → (8081) 임베딩 / (5432) DB.
  - 각 단계는 **앞 단계 태그에서 온 요청만** 허용 (예: 임베딩은 `fastapi` 태그만, DB는 `spring`·`fastapi`만).
- SSH(22)는 개발자 IP만 허용.
- 임베딩 호출은 별도 VM이므로 **내부 IP**로: `EMBEDDING_URL = "http://[임베딩_VM_내부IP]:8081"`
  - IP 고정을 위해 임베딩 VM에 **고정 내부 IP(static internal IP)** 예약 권장 (재생성 시 IP 변동 방지).

### 포트 메모
- `8081`은 특정 프로그램 기본 포트가 아니라 **임의 선택값**. TEI 도커 내부 기본 포트는 80이며 `-p 8081:80`으로 매핑. Spring의 8080과 겹치지 않게 8081 배정.

## 5. 서버별 추가 작업
- **임베딩 VM (신규)**: e2-standard-2 / 디스크 30GB / 외부 IP 없음 / 태그 `embedding` / Docker 설치 후 TEI로 Qwen3 배포
  - ⚠️ 데이터 임베딩과 질문 임베딩의 **pooling·정규화 방식 일치 필수**
  - ~~mean pooling + 접두사 방식~~ → 이후 Qwen3 공식 가이드 준수로 변경됨 (last-token pooling, 문서 무접두사, 질문만 instruct — `retrospectives/search_embedding_qwen3.md` 참고)
- **FastAPI VM**: `src/core/llm.py`의 embeddings를 Vertex AI → `http://[임베딩_내부IP]:8081` HTTP 호출로 교체 (httpx 사용). 인바운드 방화벽 추가 불필요(아웃바운드는 기본 허용).
- **Nginx VM**: 변경 없음 (임베딩과 직접 통신하지 않음, 기존 Spring·FastAPI 프록시 그대로).
- **방화벽**: 임베딩 VM에 `allow-embedding` 규칙 추가 (소스 태그 `fastapi`, 포트 8081, 대상 태그 `embedding`).
