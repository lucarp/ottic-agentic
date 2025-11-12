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
    print("Database initialized successfully")


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
    print("WebSocket connection established")

    # Session state
    db = SessionLocal()
    previous_response_id: Optional[str] = None

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

                    # Run agent and stream responses
                    async for event in run_agent(
                        user_input=content,
                        db=db,
                        previous_response_id=previous_response_id
                    ):
                        # Add timestamp to all events
                        event["timestamp"] = datetime.utcnow().isoformat()

                        # Update previous_response_id if provided
                        if event.get("type") == "response_complete":
                            previous_response_id = event.get("response_id")
                            # Don't send this internal event to frontend
                            continue

                        # Send event to frontend
                        await websocket.send_json(event)

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")
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
