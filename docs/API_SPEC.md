# GLAW AI API 명세서

## 1. QnA 질문 답변 (/api/qna) POST

# Header

```bash
Content-Type: application/json
```

# Request Body

```json
{
  "query": "캘리포니아에서 음주운전하면 어떻게 돼?",
  "country_id": 1,
  "session_id": "test-001"
}
```

# Response[200] - 성공

```json
{
  "success": true,
  "status": 200,
  "code": "SUCCESS",
  "message": "성공입니다.",
  "timestamp": "2026-03-28T12:34:56.000000+00:00",
  "result": {
    "answer": "질문에 대한 답변...",
    "related_law_id_list": [105, 209],
    "search_success": true
  }
}
```

# Response[400] - 요청 오류

```json
{
  "success": false,
  "status": 400,
  "code": "COMMON400",
  "message": "요청 처리 중 오류 : 필수 입력값(query)을 확인해주세요.",
  "timestamp": "2026-03-28T12:35:10.000000+00:00",
  "result": null
}
```

# Response[400] - 법률 검색 처리 오류

```json
{
  "success": false,
  "status": 400,
  "code": "AI_RETRIEVAL_FAILED",
  "message": "법률 검색 처리 중 오류가 발생했습니다.",
  "timestamp": "2026-03-28T12:35:15.000000+00:00",
  "result": null
}
```

# Response[404] - 국가 ID 없음

```json
{
  "success": false,
  "status": 404,
  "code": "AI_COUNTRY_NOT_FOUND",
  "message": "존재하지 않는 국가 ID입니다: 999",
  "timestamp": "2026-03-28T12:35:20.000000+00:00",
  "result": null
}
```

# Response[503] - DB 연결 오류

```json
{
  "success": false,
  "status": 503,
  "code": "AI_DB_CONNECTION_FAILED",
  "message": "데이터베이스 연결에 실패했습니다.",
  "timestamp": "2026-03-28T12:35:30.000000+00:00",
  "result": null
}
```

# Response[500] - 서버 오류

```json
{
  "success": false,
  "status": 500,
  "code": "COMMON500",
  "message": "서버 내부 오류가 발생했습니다.",
  "timestamp": "2026-03-28T12:35:40.000000+00:00",
  "result": null
}
```

## 2. 법률 비교 (/api/compare) POST

# Header

```bash
Content-Type: application/json
```

# Request Body

```json
{
  "query": "음주운전 처벌 기준 비교해줘",
  "country_id_1": 1,
  "country_id_2": 2
}
```

# Response[200] - 성공

```json
{
  "success": true,
  "status": 200,
  "code": "SUCCESS",
  "message": "비교 분석 성공입니다.",
  "timestamp": "2026-03-28T12:36:00.000000+00:00",
  "result": {
    "country_1_result": {
      "related_law_ids": [12, 55],
      "summary": "첫 번째 국가 요약..."
    },
    "country_2_result": {
      "related_law_ids": [88, 91],
      "summary": "두 번째 국가 요약..."
    },
    "compare_summary": {
      "common": "공통점...",
      "diff": "차이점..."
    }
  }
}
```

# Response[400] - 요청 오류

```json
{
  "success": false,
  "status": 400,
  "code": "COMMON400",
  "message": "요청 처리 중 오류 : 필수 입력값(query, country_id_1, country_id_2)을 확인해주세요.",
  "timestamp": "2026-03-28T12:36:15.000000+00:00",
  "result": null
}
```

# Response[400] - 비교 국가 ID 동일

```json
{
  "success": false,
  "status": 400,
  "code": "AI_COMPARE_SAME_COUNTRY",
  "message": "요청 처리 중 오류 : 두 국가 ID가 같습니다.",
  "timestamp": "2026-03-28T12:36:15.000000+00:00",
  "result": null
}
```

# Response[404] - 국가 ID 없음

```json
{
  "success": false,
  "status": 404,
  "code": "AI_COMPARE_COUNTRY_NOT_FOUND",
  "message": "존재하지 않는 국가 ID입니다: [999]",
  "timestamp": "2026-03-28T12:36:25.000000+00:00",
  "result": null
}
```

# Response[503] - DB 연결 오류

```json
{
  "success": false,
  "status": 503,
  "code": "AI_DB_CONNECTION_FAILED",
  "message": "데이터베이스 연결에 실패했습니다.",
  "timestamp": "2026-03-28T12:36:35.000000+00:00",
  "result": null
}
```

# Response[500] - AI 비교 분석 실패

```json
{
  "success": false,
  "status": 500,
  "code": "AI_COMPARE_ANALYSIS_FAILED",
  "message": "비교 분석 중 오류 발생: ...",
  "timestamp": "2026-03-28T12:36:45.000000+00:00",
  "result": null
}
```

## 3. 리스크 (/api/risk) POST

# Header

```bash
Content-Type: application/json
```

# Request Body

```json
{
  "country_id": 1,
  "travel_purpose": "tourism",
  "visa_type": "short_stay",
  "age_band": "20s"
}
```

# Response[200]

```json
{
  "success": true,
  "status": 200,
  "code": "SUCCESS",
  "message": "리스크 조회 성공입니다.",
  "timestamp": "2026-04-01T10:10:00.000000+00:00",
  "result": {
    "country_id": 1,
    "overall_risk_level": "HIGH",
    "risk_list": [
      {
        "risk_title": "전자담배 반입 금지",
        "risk_level": "HIGH",
        "risk_content": "해당 국가에서는 전자담배 반입 및 소지가 엄격히 제한되며, 위반 시 강한 제재를 받을 수 있습니다.",
        "risk_actions": [
          "출국 전 전자담배 소지 여부를 확인하세요.",
          "유사 제품(가열식 기기 포함)도 반입 금지 대상인지 확인하세요."
        ],
        "law_refs": [
          {
            "law_id": 105,
            "law_type": "PEN",
            "article_no": "23152"
          }
        ],
        "issue_refs": [
          {
            "issue_id": 301,
            "title": "현지 단속 강화 기사",
            "url": "<https://example.com/news/123>",
            "published_date": "2026-03-20"
          }
        ]
      },
      {
        "risk_title": "체류기간 초과",
        "risk_level": "MEDIUM",
        "risk_content": "비자 및 체류허가 조건을 초과하면 벌금 또는 입국 제한 조치를 받을 수 있습니다.",
        "risk_actions": [
          "입국일 기준 허용 체류일을 미리 계산하세요.",
          "연장 가능 여부를 출입국 기관 공지로 확인하세요."
        ],
        "law_refs": [
          {
            "law_id": 209,
            "law_type": "INS",
            "article_no": "24"
          }
        ],
        "issue_refs": []
      }
    ]
  }
}
```

필드 보충 설명:

- law_refs: 리스크의 법령 근거 목록 (필수)
- issue_refs: 이슈 기사 참고 목록 (선택)
    - 내부필드는 ERD 기준으로 `issue_id`, `title`, `url`, `published_date`를 사용합니다.
        - 현재 이슈 API 미구현 상태에서는 빈 배열 응답해도 정상입니다.

# Response[400] - 요청 오류

```json
{
  "success": false,
  "status": 400,
  "code": "COMMON400",
  "message": "요청 처리 중 오류 : 필수 입력값(country_id, travel_purpose, visa_type, age_band)을 확인해주세요.",
  "timestamp": "2026-04-01T10:10:10.000000+00:00",
  "result": null
}
```

# Response[400] - 법률 검색 처리 오류

```json
{
  "success": false,
  "status": 400,
  "code": "AI_RETRIEVAL_FAILED",
  "message": "법률 검색 처리 중 오류가 발생했습니다.",
  "timestamp": "2026-04-01T10:10:15.000000+00:00",
  "result": null
}
```

# Response[404] - 국가 ID 없음

```json
{
  "success": false,
  "status": 404,
  "code": "AI_COUNTRY_NOT_FOUND",
  "message": "존재하지 않는 국가 ID입니다: 999",
  "timestamp": "2026-04-01T10:10:20.000000+00:00",
  "result": null
}
```

# Response[404] - 리스크 카드 없음

```json
{
  "success": false,
  "status": 404,
  "code": "AI_RISK_NOT_FOUND",
  "message": "해당 조건에 일치하는 리스크 카드가 없습니다.",
  "timestamp": "2026-04-01T10:10:25.000000+00:00",
  "result": null
}
```

# Response[503] - DB 연결 오류

```json
{
  "success": false,
  "status": 503,
  "code": "AI_DB_CONNECTION_FAILED",
  "message": "데이터베이스 연결에 실패했습니다.",
  "timestamp": "2026-04-01T10:10:30.000000+00:00",
  "result": null
}
```

# Response[500] - 서버 오류

```json
{
  "success": false,
  "status": 500,
  "code": "COMMON500",
  "message": "서버 내부 오류가 발생했습니다.",
  "timestamp": "2026-04-01T10:10:40.000000+00:00",
  "result": null
}
```