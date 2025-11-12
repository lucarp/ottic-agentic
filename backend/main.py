"""FastAPI application with WebSocket for real-time agent communication."""

import json
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agent_streaming import run_agent_streaming as run_agent
from database import SessionLocal, init_db

# Initialize FastAPI
app = FastAPI(title="Agentic Artifact POC", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    init_db()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Agentic Artifact POC API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent communication.

    Protocol:
    - Client sends: {"type": "user_message", "content": "..."}
    - Server sends: Various message types (assistant_message, tool_execution, artifact_created, error)
    """
    await websocket.accept()

    # Session state
    db = SessionLocal()
    conversation_history = []  # Store conversation for context injection
    max_history_messages = 10  # Keep last 10 messages for context

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "assistant_message",
            "content": "Hello! I'm an AI assistant with the ability to create and manage artifacts. Ask me to analyze data, generate reports, or create any structured information.",
            "timestamp": datetime.utcnow().isoformat()
        })

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                message_type = message.get("type")
                content = message.get("content", "")

                # Echo user message back to client
                if message_type == "user_message":
                    await websocket.send_json({
                        "type": "user_message",
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Add user message to history
                    conversation_history.append({"role": "user", "content": content})

                    # Keep only last N messages
                    if len(conversation_history) > max_history_messages:
                        conversation_history = conversation_history[-max_history_messages:]

                    # Run agent and stream responses
                    assistant_response = ""
                    async for event in run_agent(
                        user_input=content,
                        db=db,
                        conversation_history=conversation_history[:-1]  # Pass history without current message
                    ):
                        # Add timestamp to all events
                        event["timestamp"] = datetime.utcnow().isoformat()

                        # Collect assistant response text
                        if event.get("type") == "text_delta":
                            assistant_response += event.get("delta", "")

                        # Skip internal events
                        if event.get("type") == "response_complete":
                            continue

                        # Send event to frontend
                        await websocket.send_json(event)

                    # Add assistant response to history
                    if assistant_response:
                        conversation_history.append({"role": "assistant", "content": assistant_response})

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
