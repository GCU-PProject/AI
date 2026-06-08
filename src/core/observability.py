"""LangSmith tracing setup helpers."""

import os

from dotenv import load_dotenv


def setup_langsmith() -> None:
    """Load LangSmith env vars and support both old/new variable names.

    LangChain/LangSmith read tracing settings from ``os.environ``. Our
    pydantic settings object reads .env values for app config, but it does not
    publish unknown keys such as LANGSMITH_* back into ``os.environ``.
    """
    load_dotenv()

    if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

    if not os.getenv("LANGSMITH_PROJECT") and os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]

    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
