from typing import List
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    AI = "AI"
    HUMAN = "Human"


class ChatRequest(BaseModel):
    user_id: str
    message: str
    image: List[str]


class ChatResponse(BaseModel):
    role: Role
    message: str
    image: List[str]


class ChatStreamResponse(BaseModel):
    type: str
    content: str
