# Production-Ready Streaming Architecture Plan

## Executive Summary

This document outlines the plan to transform the current POC into a production-ready system with real-time streaming capabilities for both LLM output and artifact creation/updates.

**Current State**: Non-streaming responses cause delays and poor UX
**Target State**: Real-time streaming of reasoning, text, tool execution, and artifacts
**Timeline**: 2-3 weeks for core streaming, 4-6 weeks for full production readiness

---

## 1. Current Architecture Limitations

### Problems with POC

1. **Blocking Responses**: Agent waits for complete GPT-5 response before sending anything
2. **No Progress Feedback**: Users see nothing until the entire response is complete
3. **Poor UX for Long Operations**: Chart/table generation feels slow
4. **No Incremental Updates**: Can't show intermediate artifact states
5. **Memory Issues**: Large responses buffered entirely in memory

### Impact

- **Latency**: 3-10 seconds before first token appears
- **User Frustration**: "Is it working?" moments
- **Scalability**: Limited concurrent users due to blocking I/O
- **Resource Waste**: Backend holds connections open unnecessarily

---

## 2. Streaming Architecture Overview

### High-Level Flow

```
User sends message
    ↓
WebSocket receives
    ↓
GPT-5 Responses API (stream=True)
    ↓
Stream chunks to frontend in real-time
    ├─→ Reasoning deltas
    ├─→ Text deltas
    ├─→ Function call deltas
    └─→ Artifact creation/updates
    ↓
Frontend renders incrementally
```

### Key Components

1. **Backend Streaming Pipeline**
   - Async generators for GPT-5 stream chunks
   - Tool execution with progress updates
   - Artifact creation with intermediate states

2. **Frontend Stream Handler**
   - Text accumulation and rendering
   - Artifact progressive loading
   - Real-time UI updates

3. **Database Optimization**
   - Async database operations
   - Connection pooling
   - Optimistic updates

---

## 3. Backend Implementation Plan

### Phase 1: Enable GPT-5 Streaming

**File**: `backend/agent.py`

#### Current Code
```python
response = await client.responses.create(
    model="gpt-5",
    input=user_input,
    stream=False,  # ❌ Blocking
    tools=TOOLS
)

# Process complete response
for item in response.output:
    if item.type == "text":
        yield {"type": "assistant_message", "content": item.content}
```

#### New Streaming Code
```python
response = await client.responses.create(
    model="gpt-5",
    input=user_input,
    stream=True,  # ✅ Streaming enabled
    tools=TOOLS
)

# Stream chunks in real-time
async for chunk in response:
    if chunk.type == "text_delta":
        # Stream text incrementally
        yield {
            "type": "text_delta",
            "delta": chunk.delta,
            "response_id": chunk.response_id
        }

    elif chunk.type == "text_done":
        # Text segment complete
        yield {
            "type": "text_done",
            "response_id": chunk.response_id
        }

    elif chunk.type == "reasoning_delta":
        # Stream reasoning
        yield {
            "type": "reasoning_delta",
            "delta": chunk.delta,
            "response_id": chunk.response_id
        }

    elif chunk.type == "function_call_delta":
        # Buffer function arguments
        tool_call_buffers[chunk.call_id]["arguments"] += chunk.delta

    elif chunk.type == "function_call_done":
        # Execute complete function call
        tool_call = tool_call_buffers[chunk.call_id]
        await execute_tool_with_streaming(tool_call, db)
```

### Phase 2: Streaming Tool Execution

**Challenge**: Tool execution can take time (database writes, validation)

**Solution**: Stream progress updates during execution

```python
async def execute_artifact_tool_streaming(
    tool_name: str,
    tool_input: dict,
    db: Session
) -> AsyncGenerator[dict, None]:
    """Execute tool with progress streaming."""

    # 1. Notify start
    yield {
        "type": "tool_execution_started",
        "tool_name": tool_name,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # 2. Validate input (can take time for complex schemas)
        yield {
            "type": "tool_execution_progress",
            "tool_name": tool_name,
            "step": "validating_input"
        }

        artifact_type, schema_class = get_artifact_mapping(tool_name)
        validated_data = schema_class(**tool_input)

        # 3. Create artifact
        yield {
            "type": "tool_execution_progress",
            "tool_name": tool_name,
            "step": "creating_artifact"
        }

        artifact = await create_artifact_async(
            db=db,
            artifact_type=artifact_type,
            data=validated_data.model_dump(),
            status=ArtifactStatus.INTERMEDIATE  # ⭐ Start as intermediate
        )

        # 4. Stream intermediate artifact
        yield {
            "type": "artifact_intermediate",
            "artifact": serialize_artifact(artifact)
        }

        # 5. Finalize artifact (any post-processing)
        yield {
            "type": "tool_execution_progress",
            "tool_name": tool_name,
            "step": "finalizing"
        }

        artifact.status = ArtifactStatus.FINAL
        await db.commit()
        await db.refresh(artifact)

        # 6. Stream final artifact
        yield {
            "type": "artifact_final",
            "artifact": serialize_artifact(artifact)
        }

        # 7. Notify completion
        yield {
            "type": "tool_execution_completed",
            "tool_name": tool_name,
            "artifact_id": str(artifact.id),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        yield {
            "type": "tool_execution_failed",
            "tool_name": tool_name,
            "error": str(e)
        }
```

### Phase 3: Async Database Operations

**File**: `backend/database.py`

**Current**: Synchronous SQLAlchemy
**Target**: Async SQLAlchemy with asyncpg driver

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)

# Async engine
async_engine = create_async_engine(
    DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Async artifact creation
async def create_artifact_async(
    db: AsyncSession,
    artifact_type: str,
    data: dict,
    status: ArtifactStatus = ArtifactStatus.FINAL
) -> Artifact:
    """Async artifact creation."""
    artifact = Artifact(
        type=artifact_type,
        status=status,
        data=data
    )
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)
    return artifact
```

### Phase 4: WebSocket Stream Management

**File**: `backend/main.py`

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async with AsyncSessionLocal() as db:
        try:
            while True:
                data = await websocket.receive_json()

                if data["type"] == "user_message":
                    # Stream all events from agent
                    async for event in run_agent_streaming(
                        user_input=data["content"],
                        db=db
                    ):
                        await websocket.send_json(event)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
```

---

## 4. Frontend Implementation Plan

### Phase 1: Text Delta Handling

**File**: `frontend/app.js`

```javascript
// Message accumulation state
const messageBuffers = {
    currentText: '',
    currentReasoning: '',
    currentMessageDiv: null,
    currentReasoningDiv: null
};

function handleMessage(message) {
    switch (message.type) {
        case 'text_delta':
            handleTextDelta(message);
            break;

        case 'text_done':
            handleTextDone(message);
            break;

        case 'reasoning_delta':
            handleReasoningDelta(message);
            break;

        case 'tool_execution_started':
            handleToolExecutionStarted(message);
            break;

        case 'tool_execution_progress':
            handleToolExecutionProgress(message);
            break;

        case 'artifact_intermediate':
            handleArtifactIntermediate(message);
            break;

        case 'artifact_final':
            handleArtifactFinal(message);
            break;
    }
}

function handleTextDelta(message) {
    // Create message div if doesn't exist
    if (!messageBuffers.currentMessageDiv) {
        messageBuffers.currentMessageDiv = document.createElement('div');
        messageBuffers.currentMessageDiv.className = 'message assistant-message';
        terminalMessages.appendChild(messageBuffers.currentMessageDiv);
    }

    // Append delta
    messageBuffers.currentText += message.delta;
    messageBuffers.currentMessageDiv.textContent = messageBuffers.currentText;

    // Auto-scroll
    scrollToBottom();
}

function handleTextDone(message) {
    // Finalize message
    messageBuffers.currentText = '';
    messageBuffers.currentMessageDiv = null;
}

function handleReasoningDelta(message) {
    if (!messageBuffers.currentReasoningDiv) {
        messageBuffers.currentReasoningDiv = document.createElement('div');
        messageBuffers.currentReasoningDiv.className = 'message reasoning-message';
        messageBuffers.currentReasoningDiv.textContent = '💭 Thinking: ';
        terminalMessages.appendChild(messageBuffers.currentReasoningDiv);
    }

    messageBuffers.currentReasoning += message.delta;
    messageBuffers.currentReasoningDiv.textContent = '💭 Thinking: ' + messageBuffers.currentReasoning;
    scrollToBottom();
}
```

### Phase 2: Tool Execution Progress

```javascript
// Track active tool executions
const activeTools = new Map();

function handleToolExecutionStarted(message) {
    const toolDiv = document.createElement('div');
    toolDiv.className = 'message tool-execution in-progress';
    toolDiv.id = `tool-${message.tool_name}-${Date.now()}`;
    toolDiv.innerHTML = `
        <div class="tool-header">
            🔧 ${message.tool_name}
            <span class="spinner">⏳</span>
        </div>
        <div class="tool-progress">Starting...</div>
    `;

    terminalMessages.appendChild(toolDiv);
    activeTools.set(message.tool_name, toolDiv);
    scrollToBottom();
}

function handleToolExecutionProgress(message) {
    const toolDiv = activeTools.get(message.tool_name);
    if (!toolDiv) return;

    const progressDiv = toolDiv.querySelector('.tool-progress');
    const stepLabels = {
        'validating_input': 'Validating input...',
        'creating_artifact': 'Creating artifact...',
        'finalizing': 'Finalizing...'
    };

    progressDiv.textContent = stepLabels[message.step] || message.step;
}

function handleToolExecutionCompleted(message) {
    const toolDiv = activeTools.get(message.tool_name);
    if (!toolDiv) return;

    toolDiv.classList.remove('in-progress');
    toolDiv.classList.add('completed');

    const spinner = toolDiv.querySelector('.spinner');
    spinner.textContent = '✅';

    activeTools.delete(message.tool_name);
}
```

### Phase 3: Progressive Artifact Rendering

```javascript
// Track artifacts being created
const artifactStates = new Map();

function handleArtifactIntermediate(message) {
    const artifact = message.artifact;

    // Show "Creating..." state
    const tab = createArtifactTab(artifact, 'intermediate');
    const viewer = createArtifactViewer(artifact, 'intermediate');

    artifactStates.set(artifact.id, {
        tab,
        viewer,
        status: 'intermediate'
    });

    // Render with loading indicator
    renderArtifactWithLoader(artifact);
}

function handleArtifactFinal(message) {
    const artifact = message.artifact;
    const state = artifactStates.get(artifact.id);

    if (state) {
        // Update existing artifact
        state.tab.classList.remove('intermediate');
        state.tab.classList.add('final');
        state.viewer.classList.remove('loading');

        // Re-render with final data
        renderArtifact(artifact, state.viewer);
    } else {
        // Create new (in case intermediate was missed)
        createAndRenderArtifact(artifact);
    }
}

function renderArtifactWithLoader(artifact) {
    const viewer = artifactStates.get(artifact.id).viewer;

    viewer.innerHTML = `
        <div class="artifact-header">
            <div class="artifact-title">${artifact.type.toUpperCase()}</div>
            <span class="status-badge intermediate">Creating...</span>
        </div>
        <div class="artifact-data loading">
            <div class="loader-spinner"></div>
            <p>Generating artifact...</p>
        </div>
    `;
}
```

---

## 5. Performance Optimizations

### 5.1 Chunking Strategy

**Problem**: Sending every character creates too many WebSocket messages

**Solution**: Batch deltas with smart chunking

```python
class DeltaBuffer:
    def __init__(self, max_size=50, max_delay=0.05):
        self.buffer = ""
        self.max_size = max_size
        self.max_delay = max_delay
        self.last_flush = time.time()

    async def add(self, delta: str) -> Optional[str]:
        self.buffer += delta

        # Flush if buffer full or time elapsed
        if len(self.buffer) >= self.max_size or \
           (time.time() - self.last_flush) >= self.max_delay:
            return await self.flush()

        return None

    async def flush(self) -> str:
        if not self.buffer:
            return None

        result = self.buffer
        self.buffer = ""
        self.last_flush = time.time()
        return result

# Usage in agent
delta_buffer = DeltaBuffer()

async for chunk in response:
    if chunk.type == "text_delta":
        batched_delta = await delta_buffer.add(chunk.delta)
        if batched_delta:
            yield {
                "type": "text_delta",
                "delta": batched_delta
            }
```

### 5.2 Connection Pooling

```python
# backend/database.py
async_engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,          # Base connections
    max_overflow=40,       # Additional connections under load
    pool_pre_ping=True,    # Test connections before use
    pool_recycle=3600,     # Recycle after 1 hour
    pool_timeout=30,       # Wait 30s for connection
    echo=False             # Disable SQL logging in production
)
```

### 5.3 Frontend Debouncing

```javascript
// Debounce scroll updates
let scrollTimeout;
function scrollToBottom() {
    if (scrollTimeout) return;

    scrollTimeout = setTimeout(() => {
        terminalMessages.scrollTop = terminalMessages.scrollHeight;
        scrollTimeout = null;
    }, 16); // ~60fps
}
```

---

## 6. Testing Strategy

### 6.1 Backend Streaming Tests

```python
# tests/test_streaming.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_streams_text_deltas():
    """Test that agent yields text deltas."""
    db = AsyncMock()

    with patch('agent.client.responses.create') as mock_create:
        # Mock streaming response
        mock_create.return_value = async_stream([
            {"type": "text_delta", "delta": "Hello "},
            {"type": "text_delta", "delta": "world"},
            {"type": "text_done"}
        ])

        events = []
        async for event in run_agent_streaming("test", db):
            events.append(event)

        # Assert we got deltas
        assert events[0]["type"] == "text_delta"
        assert events[0]["delta"] == "Hello "
        assert events[1]["type"] == "text_delta"
        assert events[1]["delta"] == "world"

@pytest.mark.asyncio
async def test_tool_execution_streams_progress():
    """Test tool execution yields progress updates."""
    db = AsyncMock()

    events = []
    async for event in execute_artifact_tool_streaming(
        "create_csv_artifact",
        {"headers": ["A"], "rows": [[1]]},
        db
    ):
        events.append(event)

    # Assert progress events
    assert events[0]["type"] == "tool_execution_started"
    assert any(e["type"] == "tool_execution_progress" for e in events)
    assert events[-1]["type"] == "tool_execution_completed"
```

### 6.2 Frontend Streaming Tests

```javascript
// tests/streaming.test.js
describe('Streaming Message Handler', () => {
    it('should accumulate text deltas', () => {
        handleMessage({ type: 'text_delta', delta: 'Hello ' });
        handleMessage({ type: 'text_delta', delta: 'world' });

        const messageDiv = document.querySelector('.assistant-message');
        expect(messageDiv.textContent).toBe('Hello world');
    });

    it('should show tool progress', () => {
        handleToolExecutionStarted({ tool_name: 'create_csv_artifact' });
        handleToolExecutionProgress({
            tool_name: 'create_csv_artifact',
            step: 'validating_input'
        });

        const toolDiv = document.querySelector('.tool-execution');
        expect(toolDiv.textContent).toContain('Validating input...');
    });
});
```

### 6.3 Load Testing

```python
# tests/load_test.py
import asyncio
import websockets

async def simulate_user():
    """Simulate single user with streaming."""
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        await ws.send(json.dumps({
            'type': 'user_message',
            'content': 'Create a chart with 100 data points'
        }))

        message_count = 0
        async for message in ws:
            message_count += 1
            # Ensure we receive streaming chunks
            assert message_count > 1  # Should receive multiple chunks

async def load_test_100_users():
    """Test with 100 concurrent users."""
    tasks = [simulate_user() for _ in range(100)]
    await asyncio.gather(*tasks)

# Run: pytest tests/load_test.py
```

---

## 7. Monitoring & Observability

### 7.1 Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Stream metrics
stream_chunks_sent = Counter(
    'stream_chunks_sent_total',
    'Total streaming chunks sent',
    ['event_type']
)

stream_latency = Histogram(
    'stream_chunk_latency_seconds',
    'Time between stream chunks',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

active_streams = Gauge(
    'active_websocket_streams',
    'Number of active streaming connections'
)

# Usage
async for chunk in response:
    stream_chunks_sent.labels(event_type=chunk.type).inc()
    yield chunk
```

### 7.2 Logging

```python
import structlog

logger = structlog.get_logger()

async def run_agent_streaming(user_input, db):
    logger.info("stream_started", user_input_length=len(user_input))

    chunk_count = 0
    start_time = time.time()

    try:
        async for chunk in response:
            chunk_count += 1
            yield chunk

    finally:
        duration = time.time() - start_time
        logger.info(
            "stream_completed",
            chunk_count=chunk_count,
            duration_seconds=duration,
            chunks_per_second=chunk_count / duration
        )
```

---

## 8. Deployment Plan

### 8.1 Phased Rollout

**Week 1-2: Core Streaming**
- Implement GPT-5 streaming
- Add text delta handling
- Basic tool execution streaming

**Week 3-4: Optimization**
- Async database operations
- Connection pooling
- Frontend performance tuning

**Week 5-6: Production Hardening**
- Load testing
- Monitoring setup
- Error handling refinement

### 8.2 Feature Flags

```python
# config.py
class Config:
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "50"))
    STREAM_CHUNK_DELAY = float(os.getenv("STREAM_CHUNK_DELAY", "0.05"))

# Usage in agent
if Config.ENABLE_STREAMING:
    response = await client.responses.create(stream=True, ...)
else:
    response = await client.responses.create(stream=False, ...)
```

### 8.3 Backward Compatibility

```javascript
// frontend/app.js - Support both modes
function handleMessage(message) {
    // New streaming events
    if (message.type === 'text_delta') {
        handleTextDelta(message);
    }
    // Legacy complete messages
    else if (message.type === 'assistant_message') {
        handleLegacyMessage(message);
    }
}
```

---

## 9. Success Metrics

### Performance Targets

| Metric | Current (POC) | Target |
|--------|--------------|--------|
| Time to First Token | 3-10 seconds | < 500ms |
| Streaming Latency | N/A | < 100ms between chunks |
| Concurrent Users | ~10 | 1000+ |
| Memory per Connection | 50-100MB | < 10MB |
| Artifact Creation Time | 2-5 seconds | < 1 second (perceived) |

### User Experience Targets

- ✅ User sees reasoning within 500ms
- ✅ Text appears character-by-character (or in small batches)
- ✅ Tool execution shows progress
- ✅ Artifacts render progressively
- ✅ No "frozen" UI states

---

## 10. Next Steps

### Immediate (Week 1)

1. ✅ Create this plan document
2. 🔄 Set up development branch for streaming work
3. 🔄 Implement basic GPT-5 streaming in agent.py
4. 🔄 Add frontend text delta handling
5. 🔄 Test with simple prompts

### Short-term (Weeks 2-3)

1. Implement tool execution streaming
2. Add async database operations
3. Create progress indicators in UI
4. Write unit tests for streaming

### Medium-term (Weeks 4-6)

1. Performance optimization (chunking, pooling)
2. Load testing with 100+ concurrent users
3. Monitoring and logging setup
4. Production deployment

### Long-term (Months 2-3)

1. Advanced features (artifact versioning during creation)
2. Real-time collaboration streaming
3. Predictive prefetching
4. Edge caching for static artifacts

---

## 11. Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPT-5 streaming API changes | High | Version pinning, integration tests |
| WebSocket connection drops | Medium | Auto-reconnect, message replay |
| Database connection pool exhaustion | High | Circuit breaker, queue overflow handling |
| Frontend memory leaks | Medium | Proper cleanup, memory profiling |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Increased cloud costs | Medium | Rate limiting, connection limits |
| Breaking changes in deployment | High | Feature flags, gradual rollout |
| Performance regression | Medium | Continuous benchmarking, alerts |

---

## Conclusion

This plan transforms the POC into a production-ready streaming platform by:

1. **Enabling real-time streaming** for all agent output
2. **Optimizing backend** with async operations and pooling
3. **Enhancing UX** with progressive rendering and progress feedback
4. **Ensuring reliability** through testing and monitoring
5. **Planning for scale** with performance optimizations

The streaming architecture will provide:
- Sub-second time to first token
- Smooth, real-time user experience
- Support for 1000+ concurrent users
- Foundation for advanced features (collaboration, versioning)

**Recommended Timeline**: 6 weeks to full production readiness
**Key Milestone**: Week 2 - Basic streaming working end-to-end
**Success Criteria**: < 500ms to first token, smooth text streaming, visible tool progress
