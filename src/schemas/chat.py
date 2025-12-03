# src/schemas/chat.py
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
