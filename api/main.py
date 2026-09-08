import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import run_agent, ConversationState


# =====================================================
# FastAPI
# =====================================================

app = FastAPI(
    title="Book RAG Assistant",
    description="A local RAG question-answering system",
)


# =====================================================
# Static Files
# =====================================================

app.mount(
    "/web",
    StaticFiles(directory="web"),
    name="web",
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Conversation Memory
# =====================================================

conversations: dict[str, ConversationState] = {}


# =====================================================
# Request / Response Models
# =====================================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    retrieved_chunks: list[str] | None = None


# =====================================================
# Web Interface
# =====================================================

@app.get("/")
async def home():
    return FileResponse("web/index.html")


# =====================================================
# Create Conversation
# =====================================================

@app.post("/conversation")
async def create_conversation():

    conversation_id = str(uuid.uuid4())

    conversations[conversation_id] = ConversationState()

    return {
        "conversation_id": conversation_id
    }


# =====================================================
# Chat
# =====================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # -------------------------------------------------
    # Get conversation ID
    # -------------------------------------------------

    conversation_id = request.conversation_id

    # -------------------------------------------------
    # Create conversation if necessary
    # -------------------------------------------------

    if conversation_id is None:

        conversation_id = str(uuid.uuid4())

        conversations[conversation_id] = ConversationState()

    elif conversation_id not in conversations:

        conversations[conversation_id] = ConversationState()

    # -------------------------------------------------
    # Get persistent conversation state
    # -------------------------------------------------

    state = conversations[conversation_id]

    # -------------------------------------------------
    # Run agent
    # -------------------------------------------------

    answer, retrieved_chunks = run_agent(
        request.message,
        state=state,
    )

    # -------------------------------------------------
    # Return answer + retrieval information
    # -------------------------------------------------

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
    )


# =====================================================
# Clear Conversation
# =====================================================

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):

    conversations.pop(
        conversation_id,
        None,
    )

    return {
        "message": "Conversation cleared"
    }
