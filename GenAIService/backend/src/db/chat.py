from uuid import UUID, uuid4
from typing import Dict, List
from enum import Enum

from beanie import Document
from pydantic import Field, BaseModel


class Message(BaseModel):
    content: str = Field(..., description="Content of the message")
    timestamp: str = Field(..., description="Timestamp of the message")
    associated_images: List[str] = Field(
        ..., description="Images associated with this message"
    )
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata for the message"
    )


class ImageKey(BaseModel):
    source_image_id: UUID = Field(..., description="Id of the Chat")
    annotation_id: UUID = Field(..., description="Id of the Chat")


class ChatMessage(BaseModel):
    human: Message = Field(..., description="Question by Human")
    ai: Message = Field(..., description="Answer by AI")
    image_key: ImageKey = Field(
        ..., description="source image id and annotation id associated with each image"
    )


class Chat(Document):
    id: UUID = Field(..., description="Id of the Chat")
    user_id: UUID = Field(..., description="id of the user associated")
    scan_id: UUID = Field(..., description="scan id associated with chat")
    messages: List[ChatMessage] = Field(
        ..., description="All messages associated with this chat"
    )

    class Settings:
        name = "chat"
