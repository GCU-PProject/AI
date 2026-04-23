# src/services/memory.py
"""
대화 기록 관리 모듈 (Conversation Memory)

이 파일은 사용자의 대화 이력을 관리하여,
후속 질문에서 이전 맥락을 참고할 수 있게 합니다.

[왜 필요한가?]
RAG 시스템은 기본적으로 각 질문을 독립적으로 처리합니다.
그래서 사용자가 "그러면 벌금은?" 같은 후속 질문을 하면,
"그러면"이 무엇을 가리키는지 알 수 없어 검색이 실패합니다.

이 모듈은 대화 기록을 보고 후속 질문을 독립적인 질문으로 재구성합니다.
예: "그러면 벌금은?" → "캘리포니아 음주운전의 벌금은 얼마인가?"

[핵심 구성 요소]
1. chat_histories (딕셔너리): session_id별 대화 기록 저장소
2. contextualize_question(): 대화 맥락을 반영하여 질문 재구성
3. save_to_history(): 질문/답변을 대화 기록에 저장

[전체 흐름]
사용자 질문 + session_id
    → get_chat_history()로 이전 대화 기록 조회
    → contextualize_question()으로 질문 재구성 (대화 기록이 있을 때만)
    → chat_service.py에서 재구성된 질문으로 RAG 파이프라인 실행
    → save_to_history()로 이번 질문/답변을 기록에 저장

[메모리 저장 방식]
현재는 Python 딕셔너리(인메모리) 방식입니다.
- 장점: 추가 인프라 없이 바로 사용 가능
- 단점: 서버를 재시작하면 모든 대화 기록이 초기화됨
- 향후: DB(PostgreSQL)에 저장하도록 이 파일만 수정하면 됨
"""

from typing import Dict, Optional
from langchain_core.prompts import load_prompt
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# 1. 대화 기록 저장소
# =========================================================
# session_id를 키(key)로, 대화 기록 객체를 값(value)으로 저장하는 딕셔너리입니다.
#
# [구조 예시]
# {
#     "abc-123": ChatMessageHistory([Human("음주운전 처벌?"), AI("VEH 23152에 따르면...")]),
#     "def-456": ChatMessageHistory([Human("교통사고 보상?"), AI("CIV 3333에 따르면...")]),
# }
#
# ChatMessageHistory: LangChain에서 제공하는 대화 기록 관리 클래스
# - 내부에 messages 리스트를 가지고 있음
# - add_user_message(), add_ai_message()로 메시지 추가
chat_histories: Dict[str, ChatMessageHistory] = {}

# 슬라이딩 윈도우 크기: 질문 재구성 시 참고할 최근 대화 쌍 수
# - 3쌍(6개 메시지) = 직전 3번의 질문/답변만 참고
# - 너무 크면: 오래된 주제가 섞여 재구성 품질 저하
# - 너무 작으면: 맥락 유지 부족
MEMORY_WINDOW_SIZE = 3


def get_chat_history(session_id: Optional[str]) -> ChatMessageHistory:
    """
    session_id에 해당하는 대화 기록을 반환합니다.
    처음 요청하는 session_id면 빈 대화 기록을 새로 생성합니다.

    Args:
        session_id: 대화 세션 고유 ID (프론트엔드에서 UUID로 생성)

    Returns:
        ChatMessageHistory 객체 (messages 리스트 포함)
    """
    if session_id is None:
        return ChatMessageHistory()

    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


# =========================================================
# 2. 질문 재구성 (Contextualize Question)
# =========================================================
# [핵심 아이디어]
# 후속 질문을 대화 기록을 참고하여 독립적인 질문으로 재구성합니다.
# 재구성된 질문은 벡터 검색에만 사용되고,
# 최종 LLM 답변 생성에는 원본 질문이 사용됩니다.
#
# [예시]
# 대화 기록: [("음주운전 처벌이 뭐야?", "VEH 23152에 따르면...")]
# 후속 질문: "그러면 벌금은?"
# 재구성 결과: "캘리포니아 음주운전(DUI)의 벌금은 얼마인가?"
#
# [왜 별도 프롬프트가 필요한가?]
# 대화 기록을 그대로 검색 쿼리에 넣으면 너무 길어지고,
# 벡터 검색 정확도가 떨어집니다.
# 대신 LLM에게 "핵심만 뽑아서 검색용 질문으로 바꿔줘"라고 요청합니다.

# 질문 재구성용 프롬프트
# - few-shot 예시를 포함하여 LLM이 답변 대신 짧은 질문만 출력하도록 유도
# - MessagesPlaceholder("chat_history"): 대화 기록이 이 자리에 삽입됨
# - {input}: 사용자의 최신 질문이 삽입됨

contextualize_yaml = load_prompt("src/prompts/contextualize.yaml", encoding="utf-8")

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_yaml.template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


async def contextualize_question(query: str, session_id: Optional[str], llm) -> str:
    """
    대화 기록을 참고하여 후속 질문을 독립적인 질문으로 재구성합니다.

    대화 기록이 비어있으면 (첫 질문이면) 원본 질문을 그대로 반환합니다.

    Args:
        query: 사용자의 원본 질문
        session_id: 대화 세션 ID
        llm: LLM 객체 (chat_service.py의 llm을 전달받음)

    Returns:
        재구성된 질문 문자열 (또는 첫 질문이면 원본 그대로)

    [동작 예시]
    첫 질문: "음주운전 처벌?" → 대화 기록 없음 → "음주운전 처벌?" (그대로)
    후속:    "벌금은?"       → 대화 기록 참고 → "캘리포니아 음주운전 벌금?" (재구성)
    """
    if session_id is None:
        return query

    history = get_chat_history(session_id)

    # 대화 기록이 비어있으면 (첫 질문) 재구성할 필요 없음
    if not history.messages:
        return query

    # LCEL 체인: 프롬프트 → LLM → 텍스트 추출
    chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    # 저장 시점에 이미 MEMORY_WINDOW_SIZE 만큼 잘려있으므로 전체를 그대로 전달합니다.
    recent_messages = history.messages

    contextualized = await chain.ainvoke(
        {
            "chat_history": recent_messages,
            "input": query,
        }
    )

    # 안전장치: 재구성 결과가 비어있으면 원본 질문을 그대로 사용
    # (토큰 제한 등으로 LLM이 빈 문자열을 반환하는 경우 방지)
    if not contextualized or not contextualized.strip():
        print(f"⚠️ 질문 재구성 결과가 비어있어 원본 질문을 사용합니다: '{query}'")
        return query

    print(f"💬 질문 재구성: '{query}' → '{contextualized}'")
    return contextualized


# =========================================================
# 3. 대화 기록 저장
# =========================================================


def save_to_history(session_id: Optional[str], query: str, answer: str) -> None:
    """
    질문과 답변을 대화 기록에 저장합니다.

    LangChain의 ChatMessageHistory는 Human/AI 메시지를 구분하여 저장합니다.
    이 기록은 다음 질문의 contextualize_question()에서 참고됩니다.

    Args:
        session_id: 대화 세션 ID
        query: 사용자의 원본 질문
        answer: AI가 생성한 답변
    """
    if session_id is None:
        return

    history = get_chat_history(session_id)
    history.add_user_message(query)
    history.add_ai_message(answer)
    
    # [메모리 누수 방지]
    # 사용자가 수백 번 질문하면 상자(RAM)가 무한히 커지는 것을 방지하기 위해
    # 오래된 대화 기록은 잘라내고 최근 기록(MEMORY_WINDOW_SIZE * 2)만 유지합니다.
    max_messages = MEMORY_WINDOW_SIZE * 2
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]
        
    print(
        f"💾 대화 기록 저장 (session: {session_id}, 총 {len(history.messages)}개 메시지 유지)"
    )
