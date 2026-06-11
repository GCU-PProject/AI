# src/services/memory.py
"""
대화 기록 관리 모듈 (Conversation Memory)

후속 질문("그러면 벌금은?")을 대화 기록 기반으로 독립 질문
("캘리포니아 음주운전 벌금은?")으로 재구성해 검색이 성립하게 한다.

저장 방식: 인메모리 딕셔너리 (서버 재시작 시 초기화됨,
DB 저장이 필요해지면 이 파일만 수정하면 됨)
"""

import logging
from typing import Dict, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt

logger = logging.getLogger(__name__)

# =========================================================
# 1. 대화 기록 저장소 — {session_id: ChatMessageHistory}
# =========================================================
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
# 재구성된 질문은 벡터 검색에만 사용되고, 답변 생성에는 원본 질문이 사용된다.
# 대화 기록을 검색 쿼리에 그대로 넣으면 길어져 검색 정확도가 떨어지므로,
# LLM이 핵심만 뽑은 짧은 독립 질문으로 바꾼다. (few-shot 예시로 형식 유도)

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
    대화 기록이 비어있으면(첫 질문) 원본 질문을 그대로 반환합니다.

    Args:
        query: 사용자의 원본 질문
        session_id: 대화 세션 ID
        llm: LLM 객체 (chat_service.py의 llm을 전달받음)
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

    contextualized_res = await chain.ainvoke(
        {
            "chat_history": recent_messages,
            "input": query,
        }
    )
    contextualized = str(contextualized_res)


    # 안전장치: 재구성 결과가 비어있으면 원본 질문을 그대로 사용
    # (토큰 제한 등으로 LLM이 빈 문자열을 반환하는 경우 방지)
    if not contextualized or not contextualized.strip():
        logger.warning(
            "⚠️ 질문 재구성 결과가 비어있어 원본 질문을 사용합니다: '%s'", query
        )
        return query

    logger.info("💬 질문 재구성: '%s' → '%s'", query, contextualized)
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

    logger.info(
        "💾 대화 기록 저장 (session: %s, 총 %s개 메시지 유지)",
        session_id,
        len(history.messages),
    )
