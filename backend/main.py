"""FastAPI application with WebSocket for real-time agent communication."""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agent_streaming import run_agent_streaming as run_agent
from database import SessionLocal, init_db

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info("✅ WebSocket connection accepted")

    # Session state
    db = SessionLocal()
    conversation_history = []  # Store conversation for context injection
    max_history_messages = 10  # Keep last 10 messages for context
    logger.info(f"📝 Session initialized with max_history={max_history_messages}")

    try:
        # Send welcome message
        welcome_msg = {
            "type": "assistant_message",
            "content": "Hello! I'm an AI assistant with the ability to create and manage artifacts. Ask me to analyze data, generate reports, or create any structured information.",
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_json(welcome_msg)
        logger.info("👋 Welcome message sent")

        while True:
            # Receive message from client
            logger.debug("⏳ Waiting for client message...")
            data = await websocket.receive_text()
            logger.info(f"📨 Received raw data: {data[:100]}...")

            try:
                message = json.loads(data)
                message_type = message.get("type")
                content = message.get("content", "")
                logger.info(f"📩 Parsed message type='{message_type}', content_length={len(content)}")

                # Echo user message back to client
                if message_type == "user_message":
                    logger.info(f"💬 User message: '{content[:50]}...'")

                    echo_msg = {
                        "type": "user_message",
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await websocket.send_json(echo_msg)
                    logger.info("✅ Echoed user message back to client")

                    # Add user message to history
                    conversation_history.append({"role": "user", "content": content})
                    logger.info(f"📝 Added to history (total messages: {len(conversation_history)})")

                    # Keep only last N messages
                    if len(conversation_history) > max_history_messages:
                        conversation_history = conversation_history[-max_history_messages:]
                        logger.info(f"🔄 Trimmed history to {len(conversation_history)} messages")

                    # Run agent and stream responses
                    logger.info("🤖 Invoking agent with streaming...")
                    assistant_response = ""
                    event_count = 0

                    async for event in run_agent(
                        user_input=content,
                        db=db,
                        conversation_history=conversation_history[:-1]  # Pass history without current message
                    ):
                        event_count += 1
                        event_type = event.get("type")

                        # Add timestamp to all events
                        event["timestamp"] = datetime.utcnow().isoformat()

                        # Collect assistant response text
                        if event_type == "text_delta":
                            delta = event.get("delta", "")
                            assistant_response += delta
                            if event_count % 10 == 0:  # Log every 10th delta
                                logger.debug(f"📝 Text delta #{event_count} (total length: {len(assistant_response)})")

                        # Log important events
                        if event_type == "tool_execution":
                            logger.info(f"🔧 Tool execution: {event.get('tool_name')} - {event.get('status')}")
                        elif event_type == "artifact_created":
                            artifact_id = event.get("artifact", {}).get("id")
                            artifact_type = event.get("artifact", {}).get("type")
                            logger.info(f"🎨 Artifact created: type={artifact_type}, id={artifact_id}")

                        # Skip internal events
                        if event_type == "response_complete":
                            logger.info(f"✅ Agent response complete (events: {event_count}, response_length: {len(assistant_response)})")
                            continue

                        # Send event to frontend
                        await websocket.send_json(event)

                    # Add assistant response to history
                    if assistant_response:
                        conversation_history.append({"role": "assistant", "content": assistant_response})
                        logger.info(f"💾 Saved assistant response to history (length: {len(assistant_response)})")

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Error processing message: {type(e).__name__}: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {type(e).__name__}: {e}", exc_info=True)
    finally:
        db.close()
        logger.info("🔒 Database session closed")


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
