import json
import asyncio
from typing import List

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from web_api import ChatRequest, ChatResponse, Role, ChatStreamResponse
from genai_model import ChatSession

app = FastAPI(root_path="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # You can restrict this to ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for chat messages and images
chat_history = {}


async def stream_chat(request: ChatRequest):
    chat_session = chat_history[request.user_id]
    stream_response, returned_images = chat_session.generate_output(
        request.message, request.image
    )

    # Send the images first
    for image in returned_images:
        response = ChatStreamResponse(type="image", content=image)
        message = json.dumps(dict(response)) + "\n"
        yield message

    # Stream the text
    total_message = ""
    for content in stream_response:
        word = content.text
        response = ChatStreamResponse(type="text", content=word)
        message = json.dumps(dict(response)) + "\n"
        total_message += word

        yield message

    if request.user_id not in chat_history:
        raise Exception("User should be present when StreamingResponseg")

    chat_session.add_message(
        ChatResponse(role=Role.AI, message=total_message, image=returned_images)
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    if request.user_id not in chat_history:
        chat_history[request.user_id] = ChatSession()

    chat_history[request.user_id].add_message(
        ChatResponse(role=Role.HUMAN, message=request.message, image=request.image)
    )
    return StreamingResponse(stream_chat(request), media_type="application/json")


@app.get("/history", response_model=List[ChatResponse])
async def get_history(user_id: str):
    if user_id in chat_history:
        return chat_history[user_id].message_history()
    return []


@app.get("/clear", response_model=bool)
async def clear(user_id: str):
    if user_id in chat_history:
        del chat_history[user_id]
        return True
    return False
