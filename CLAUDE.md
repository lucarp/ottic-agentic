# CLAUDE.md - Ottic Agentic Platform

## Project Overview

Ottic Agentic is a production-ready platform that enables AI agents to create and manage typed artifacts through a real-time WebSocket interface. Unlike traditional chatbots that only return text, this platform allows agents to generate structured, validated data objects (artifacts) that the frontend can intelligently render and interact with.

### Core Concept

The platform introduces a **typed artifact system** where:
- Each artifact type has a strict Pydantic schema for validation
- Each artifact type is exposed as a dedicated LLM tool
- The frontend has specialized renderers for each artifact type
- All artifacts are persisted in PostgreSQL for user collections
- Real-time streaming of both agent output and artifact creation

### Key Innovation

Instead of having the LLM output unstructured JSON or markdown, we provide **specific tools per artifact type**. This ensures:
1. **Type Safety**: Pydantic validates all artifact data before storage
2. **Semantic Clarity**: The LLM knows exactly what each tool does
3. **Frontend Intelligence**: The frontend knows how to render each type
4. **Extensibility**: New artifact types are just new tools + renderers

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  WebSocket   │  │   Terminal   │  │  Artifact    │      │
│  │   Client     │  │   Display    │  │  Renderers   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────┬───────────────────────────────┘
                              │ WebSocket (JSON)
┌─────────────────────────────┴───────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  WebSocket   │  │  GPT-5 Agent │  │  Artifact    │      │
│  │   Handler    │  │  (Responses) │  │  Tools       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────┬───────────────────────────────┘
                              │ SQL
┌─────────────────────────────┴───────────────────────────────┐
│                      PostgreSQL 17                           │
│                   Artifacts Table (JSONB)                    │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

#### Backend
- **FastAPI** (0.115.0+): ASGI web framework with WebSocket support
- **OpenAI Python SDK** (1.55.0+): GPT-5 Responses API integration
- **SQLAlchemy** (2.0.0+): PostgreSQL ORM
- **Pydantic** (2.9.0+): Schema validation and serialization
- **Uvicorn**: ASGI server with WebSocket support
- **PostgreSQL 17**: Primary database with JSONB support

#### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Chart.js** (4.4.0): Data visualizations
- **Marked.js** (11.0.0): Markdown rendering
- **Highlight.js** (11.9.0): Code syntax highlighting

#### Infrastructure
- **Docker Compose**: PostgreSQL containerization
- **WebSockets**: Real-time bidirectional communication

---

## Current Artifact Types

Each artifact type is a first-class entity with:
1. Pydantic schema in `backend/schemas.py`
2. Dedicated tool in `backend/agent_agentic.py`
3. Specialized renderer in `frontend/app.js`

| Type | Schema | Use Case | Frontend Renderer |
|------|--------|----------|-------------------|
| `csv` | `CsvArtifact` | Tabular data, datasets | HTML table with styling |
| `chart` | `ChartArtifact` | Data visualizations | Chart.js canvas |
| `code` | `CodeArtifact` | Code snippets | Highlight.js syntax highlighting |
| `markdown` | `MarkdownArtifact` | Documents, reports | Marked.js rendering |
| `html` | `HtmlArtifact` | Rich content | Sandboxed rendering |
| `payment_link` | `PaymentLinkArtifact` | Payment requests | Stripe-like UI (mock) |
| `domain_overview` | `DomainOverviewArtifact` | SEO domain metrics | Metric cards with traffic/keyword data |
| `competitor_analysis` | `CompetitorAnalysisArtifact` | SEO competitor insights | Table of competing domains |
| `keyword_research` | `KeywordResearchArtifact` | Keyword opportunities | Color-coded keyword table |

---

## SE Ranking API Integration

The platform integrates with the **SE Ranking Data API** to provide professional SEO analysis capabilities. This enables the agent to deliver data-driven SEO insights without requiring the user to manually fetch data.

### SEO Capabilities

**1. Domain SEO Overview** (`analyze_domain_seo` tool)
- Provides comprehensive domain metrics including:
  - Total organic keywords ranking
  - Estimated monthly organic traffic
  - Organic traffic value (monetary estimate)
  - Paid search keywords
  - Paid traffic estimates
  - Traffic value by currency
- Creates `DomainOverviewArtifact` with metric cards visualization

**2. Competitor Analysis** (`analyze_competitors` tool)
- Identifies top organic or paid search competitors
- Shows keyword overlap with target domain
- Supports regional databases (US, UK, CA, DE, etc.)
- Creates `CompetitorAnalysisArtifact` with ranked competitor table

**3. Keyword Research** (`research_keywords` tool)
- **Similar Keywords**: Find semantically related keywords to expand content
- **Keyword Gap Analysis**: Identify keywords competitors rank for that you don't
- Includes search volume, CPC, and difficulty scores
- Creates `KeywordResearchArtifact` with color-coded difficulty ratings

### API Configuration

**Environment Variable:**
```bash
SERANKING_API_KEY=your_api_key_here
```

**API Client:** `backend/seranking_client.py`
- Async HTTP client using `httpx`
- Built-in rate limiting (10 req/sec)
- Error handling and retry logic
- Methods:
  - `get_domain_overview(domain, currency)`
  - `get_competitors(domain, source, type, limit)`
  - `get_keyword_comparison(domain, competitor, source)`
  - `get_similar_keywords(keyword, source, limit)`

### Example Usage

**Domain Analysis:**
```
User: "Analyze the SEO performance of stripe.com"
Agent: [Uses analyze_domain_seo tool]
Result: Domain overview artifact with traffic metrics
```

**Competitor Research:**
```
User: "Who are the top competitors for shopify.com?"
Agent: [Uses analyze_competitors tool with source="us"]
Result: Competitor analysis artifact with top 10 competing domains
```

**Keyword Opportunities:**
```
User: "Find keywords similar to 'payment gateway'"
Agent: [Uses research_keywords with analysis_type="similar"]
Result: Keyword research artifact with 50+ related keywords
```

**Competitive Keyword Gap:**
```
User: "What keywords does paypal.com rank for that stripe.com doesn't?"
Agent: [Uses research_keywords with analysis_type="gap"]
Result: Keyword research artifact showing opportunity keywords
```

### API Cost Considerations

- Each API call consumes SE Ranking API credits
- Approximate costs per request:
  - Domain overview: ~50 credits
  - Competitor analysis: ~100 credits
  - Keyword research: ~10-50 credits (varies by results)
- Rate limit: 10 requests per second (enforced by client)
- Monitor usage via SE Ranking dashboard

---

## File Structure

```
ottic-agentic/
├── docker-compose.yml          # PostgreSQL container
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # User-facing documentation
├── CLAUDE.md                   # This file (developer guide)
├── backend/
│   ├── main.py                 # FastAPI app + WebSocket endpoint
│   ├── agent_agentic.py        # GPT-5 agent with tool execution
│   ├── database.py             # SQLAlchemy models and helpers
│   ├── schemas.py              # Pydantic artifact schemas
│   ├── seranking_client.py     # SE Ranking API client
│   └── requirements.txt        # Python dependencies
└── frontend/
    ├── index.html              # Split-screen UI
    └── app.js                  # WebSocket client + artifact renderers
```

---

## Database Schema

### Artifacts Table

```sql
CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR NOT NULL,                    -- Artifact type (csv, chart, etc.)
    status VARCHAR NOT NULL,                  -- 'intermediate' or 'final'
    data JSONB NOT NULL,                      -- Artifact payload (validated by Pydantic)
    artifact_metadata JSONB,                  -- Tool metadata, user info, etc.
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_artifacts_type ON artifacts(type);
CREATE INDEX idx_artifacts_created_at ON artifacts(created_at DESC);
CREATE INDEX idx_artifacts_data ON artifacts USING GIN(data);
```

### Why JSONB?

- **Flexible**: Each artifact type has different data structure
- **Queryable**: Can query nested fields with PostgreSQL's JSONB operators
- **Validated**: Pydantic ensures data integrity before storage
- **Performant**: GIN indexes for fast JSON queries

---

## WebSocket Protocol

### Client → Server

```json
{
  "type": "user_message",
  "content": "Create a bar chart showing Q1-Q4 sales"
}
```

### Server → Client

#### 1. User Message Echo
```json
{
  "type": "user_message",
  "content": "Create a bar chart...",
  "timestamp": "2025-01-10T12:00:00.000Z"
}
```

#### 2. Reasoning (GPT-5)
```json
{
  "type": "reasoning",
  "content": "The user wants a bar chart. I need to use create_chart_artifact with quarterly data.",
  "response_id": "resp_abc123"
}
```

#### 3. Assistant Message
```json
{
  "type": "assistant_message",
  "content": "I'll create a bar chart showing your quarterly sales data.",
  "response_id": "resp_abc123"
}
```

#### 4. Tool Execution Start
```json
{
  "type": "tool_execution",
  "tool_name": "create_chart_artifact",
  "status": "started",
  "input": {
    "chart_type": "bar",
    "labels": ["Q1", "Q2", "Q3", "Q4"],
    "datasets": [...]
  }
}
```

#### 5. Tool Execution Complete
```json
{
  "type": "tool_execution",
  "tool_name": "create_chart_artifact",
  "status": "completed",
  "output": {
    "success": true,
    "artifact_id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "chart"
  }
}
```

#### 6. Artifact Created
```json
{
  "type": "artifact_created",
  "artifact": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "chart",
    "status": "final",
    "data": {
      "chart_type": "bar",
      "labels": ["Q1", "Q2", "Q3", "Q4"],
      "datasets": [...]
    },
    "artifact_metadata": {"tool": "create_chart_artifact"},
    "created_at": "2025-01-10T12:00:01.000Z"
  }
}
```

#### 7. Error (if any)
```json
{
  "type": "error",
  "error": "Validation failed: amount must be greater than 0"
}
```

---

## Agent Flow

### GPT-5 Responses API Integration

The agent uses OpenAI's Responses API with these key features:

1. **Conversation Context**: `previous_response_id` maintains multi-turn state
2. **Reasoning**: Medium effort reasoning for better tool selection
3. **Tool Calling**: Structured function calling with Pydantic schemas
4. **Non-streaming**: POC uses non-streaming for simplicity (production should stream)

### Execution Loop

```python
async def run_agent(user_input, db, previous_response_id):
    # 1. Create initial response
    response = await client.responses.create(
        model="gpt-5",
        input=user_input,
        reasoning={"effort": "medium"},
        tools=TOOLS,
        previous_response_id=previous_response_id
    )

    # 2. Process response items
    while True:
        for item in response.output:
            if item.type == "reasoning":
                # Stream reasoning to frontend
                yield {"type": "reasoning", "content": item.summary}

            elif item.type == "text":
                # Stream assistant message
                yield {"type": "assistant_message", "content": item.content}

            elif item.type == "function_call":
                # Execute tool
                result = execute_artifact_tool(item.name, item.arguments, db)

                # Stream tool execution status
                yield {"type": "tool_execution", "status": "completed"}

                # Stream artifact created
                if result["success"]:
                    artifact = get_artifact_by_id(db, result["artifact_id"])
                    yield {"type": "artifact_created", "artifact": artifact}

        # 3. If no more function calls, exit loop
        if not has_function_calls:
            break

        # 4. Submit tool outputs and continue
        response = await client.responses.create(
            input=tool_outputs,
            previous_response_id=response_id
        )
```

---

## Adding New Artifact Types

### Example: Adding a `TableArtifact`

#### 1. Define Schema (`backend/schemas.py`)

```python
class TableArtifact(BaseModel):
    """Interactive table with sorting and filtering."""
    headers: list[str] = Field(description="Column headers")
    rows: list[list[Any]] = Field(description="Table rows")
    sortable: bool = Field(default=True, description="Enable sorting")
    filterable: bool = Field(default=False, description="Enable filtering")
    title: Optional[str] = Field(default=None, description="Table title")
```

#### 2. Add Tool (`backend/agent.py`)

```python
TOOLS.append({
    "type": "function",
    "name": "create_table_artifact",
    "description": "Creates an interactive table with sorting and filtering capabilities.",
    "parameters": TableArtifact.model_json_schema()
})

# Update artifact_mapping in execute_artifact_tool
artifact_mapping = {
    # ... existing mappings ...
    "create_table_artifact": ("table", TableArtifact),
}
```

#### 3. Add Renderer (`frontend/app.js`)

```javascript
ArtifactRenderers.table = (data) => {
    const { headers, rows, title, sortable, filterable } = data;

    let html = title ? `<h3>${title}</h3>` : '';
    html += '<table class="interactive-table">';
    html += '<thead><tr>';

    headers.forEach(header => {
        html += `<th onclick="sortTable('${header}')">${header}`;
        if (sortable) html += ' ↕️';
        html += '</th>';
    });

    html += '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table>';

    return html;
};
```

---

## Production Readiness Plan

### Phase 1: Streaming Architecture (HIGH PRIORITY)

**Current Issue**: POC uses non-streaming responses, causing delays.

**Solution**: Implement streaming for both LLM output and artifact creation.

#### Backend Changes

```python
# In agent.py - Enable streaming
response = await client.responses.create(
    model="gpt-5",
    input=user_input,
    stream=True,  # Enable streaming
    tools=TOOLS
)

# Stream response chunks
async for chunk in response:
    if chunk.type == "text_delta":
        yield {
            "type": "text_delta",
            "delta": chunk.delta,
            "response_id": chunk.response_id
        }
    elif chunk.type == "function_call_delta":
        # Buffer function call arguments
        tool_call_buffer[chunk.call_id] += chunk.delta
```

#### Frontend Changes

```javascript
// In app.js - Handle text deltas
let currentMessageDiv = null;

function handleMessage(message) {
    if (message.type === 'text_delta') {
        if (!currentMessageDiv) {
            currentMessageDiv = document.createElement('div');
            currentMessageDiv.className = 'message assistant-message';
            terminalMessages.appendChild(currentMessageDiv);
        }
        currentMessageDiv.textContent += message.delta;
        scrollToBottom();
    }
    else if (message.type === 'assistant_message') {
        currentMessageDiv = null;  // Reset for next message
    }
}
```

### Phase 2: Authentication & Multi-User Support

**Requirements**:
- User accounts and sessions
- Artifact ownership and permissions
- Workspace/project organization

**Database Schema Changes**:

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR UNIQUE NOT NULL,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    conversation_history JSONB,  -- Store response_ids
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

ALTER TABLE artifacts ADD COLUMN user_id UUID REFERENCES users(id);
ALTER TABLE artifacts ADD COLUMN session_id UUID REFERENCES sessions(id);
```

**Backend Changes**:

```python
# Add JWT authentication middleware
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    # Verify JWT and return user
    pass

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    user = await get_current_user(token)
    # ... rest of WebSocket logic
```

### Phase 3: Artifact Versioning & Updates

**Requirements**:
- Update existing artifacts (intermediate → final)
- Version history for artifacts
- Diff visualization

**Database Schema Changes**:

```sql
CREATE TABLE artifact_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id),
    version INT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(artifact_id, version)
);

ALTER TABLE artifacts ADD COLUMN version INT DEFAULT 1;
```

**New Tool**:

```python
{
    "type": "function",
    "name": "update_artifact",
    "description": "Updates an existing artifact with new data",
    "parameters": {
        "artifact_id": "UUID of the artifact to update",
        "updates": "Partial or complete updated data"
    }
}
```

### Phase 4: Artifact Export & Sharing

**Features**:
- Export artifacts (CSV download, PDF generation)
- Public sharing links
- Embed codes for artifacts

**Implementation**:

```python
@app.get("/artifacts/{artifact_id}/export")
async def export_artifact(artifact_id: UUID, format: str = "json"):
    artifact = get_artifact_by_id(db, artifact_id)

    if format == "csv" and artifact.type == "csv":
        return generate_csv_file(artifact.data)
    elif format == "pdf":
        return generate_pdf(artifact)
    else:
        return JSONResponse(artifact.data)

@app.get("/share/{share_token}")
async def view_shared_artifact(share_token: str):
    # Return read-only artifact viewer
    pass
```

### Phase 5: Advanced Artifact Types

**New Types to Add**:

1. **Diagram Artifacts** (Mermaid.js)
   - Flowcharts, sequence diagrams, ER diagrams

2. **Interactive Forms**
   - Data collection, surveys

3. **Map Artifacts** (Leaflet.js)
   - Geographic visualizations

4. **Calendar Artifacts**
   - Event timelines, schedules

5. **3D Visualizations** (Three.js)
   - 3D charts, models

### Phase 6: Real-time Collaboration

**Features**:
- Multiple users viewing same session
- Cursor presence indicators
- Artifact co-editing

**Technology**:
- WebSocket rooms (Socket.io or custom)
- CRDT for conflict resolution
- Presence awareness protocol

### Phase 7: Production Infrastructure

#### Deployment Architecture

```
┌─────────────────┐
│   Load Balancer │ (AWS ALB / Cloudflare)
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬──────────┐
    │          │          │          │
┌───┴───┐  ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
│Backend│  │Backend│  │Backend│  │Backend│  (Auto-scaling)
└───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
    │          │          │          │
    └──────────┴──────────┴──────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────┴─────┐      ┌───────┴────┐
    │PostgreSQL│      │    Redis   │
    │ (RDS/DO) │      │   (Cache)  │
    └──────────┘      └────────────┘
```

#### Environment Variables

```bash
# Production .env
DATABASE_URL=postgresql://user:pass@prod-db:5432/ottic
REDIS_URL=redis://prod-redis:6379
OPENAI_API_KEY=sk-prod-key
JWT_SECRET=production-secret
CORS_ORIGINS=https://app.ottic.com
LOG_LEVEL=info
SENTRY_DSN=https://...
```

#### Monitoring & Observability

```python
# Add to backend/main.py
import sentry_sdk
from prometheus_client import Counter, Histogram

# Metrics
artifact_created_counter = Counter(
    'artifacts_created_total',
    'Total artifacts created',
    ['type']
)

agent_latency = Histogram(
    'agent_response_latency_seconds',
    'Agent response time'
)

# Sentry integration
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,
)
```

#### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.websocket("/ws")
@limiter.limit("100/minute")
async def websocket_endpoint(websocket: WebSocket):
    # ... existing code
```

#### Caching Strategy

```python
import redis

redis_client = redis.from_url(os.getenv("REDIS_URL"))

# Cache artifact schemas
@lru_cache
def get_artifact_schema(artifact_type: str):
    return ARTIFACT_SCHEMAS[artifact_type]

# Cache frequently accessed artifacts
def get_artifact_cached(artifact_id: UUID):
    cached = redis_client.get(f"artifact:{artifact_id}")
    if cached:
        return json.loads(cached)

    artifact = get_artifact_by_id(db, artifact_id)
    redis_client.setex(
        f"artifact:{artifact_id}",
        3600,  # 1 hour TTL
        json.dumps(artifact)
    )
    return artifact
```

---

## Security Considerations

### 1. Input Validation

- All artifact data validated by Pydantic schemas
- SQL injection prevented by SQLAlchemy ORM
- XSS prevented by frontend sanitization

### 2. Authentication

- Implement JWT-based authentication
- Secure WebSocket connections with token validation
- Rate limiting per user

### 3. Database Security

- Row-level security (RLS) for multi-tenancy
- Encrypted connections (SSL/TLS)
- Regular backups and point-in-time recovery

### 4. API Key Management

- Never expose OpenAI API keys to frontend
- Rotate keys regularly
- Use environment variables, never hardcode

### 5. CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS").split(","),  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_schemas.py
def test_csv_artifact_validation():
    valid_data = {
        "headers": ["Name", "Age"],
        "rows": [["Alice", 30], ["Bob", 25]]
    }
    artifact = CsvArtifact(**valid_data)
    assert artifact.headers == ["Name", "Age"]

def test_csv_artifact_invalid():
    with pytest.raises(ValidationError):
        CsvArtifact(headers="not a list", rows=[])
```

### Integration Tests

```python
# tests/test_agent.py
@pytest.mark.asyncio
async def test_agent_creates_csv():
    db = SessionLocal()
    events = []

    async for event in run_agent("Create a table of fruits", db):
        events.append(event)

    artifact_events = [e for e in events if e["type"] == "artifact_created"]
    assert len(artifact_events) == 1
    assert artifact_events[0]["artifact"]["type"] == "csv"
```

### E2E Tests

```javascript
// tests/e2e/test_websocket.spec.js
describe('WebSocket Artifact Creation', () => {
    it('should create and render a chart artifact', async () => {
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.send(JSON.stringify({
            type: 'user_message',
            content: 'Create a bar chart of sales'
        }));

        const messages = await waitForMessages(ws, 5);
        const artifactCreated = messages.find(m => m.type === 'artifact_created');

        expect(artifactCreated).toBeDefined();
        expect(artifactCreated.artifact.type).toBe('chart');
    });
});
```

---

## Performance Optimization

### Database Indexes

```sql
-- Query artifacts by type frequently
CREATE INDEX idx_artifacts_type ON artifacts(type);

-- Query recent artifacts
CREATE INDEX idx_artifacts_created_at ON artifacts(created_at DESC);

-- Full-text search on artifact data
CREATE INDEX idx_artifacts_data_gin ON artifacts USING GIN(data);

-- User artifacts (once multi-user is added)
CREATE INDEX idx_artifacts_user_id ON artifacts(user_id);
```

### Connection Pooling

```python
# In database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Test connections before use
    pool_recycle=3600    # Recycle connections every hour
)
```

### Async Database Operations

```python
# Upgrade to async SQLAlchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

async_engine = create_async_engine(DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))

async def create_artifact_async(db, artifact_type, data):
    async with db() as session:
        artifact = Artifact(type=artifact_type, data=data)
        session.add(artifact)
        await session.commit()
        return artifact
```

---

## Development Workflow

### Local Development

```bash
# 1. Start database
docker-compose up -d

# 2. Install dependencies
cd backend && pip install -r requirements.txt

# 3. Run backend (with hot reload)
uvicorn main:app --reload --port 8000

# 4. Open frontend
cd frontend && python -m http.server 3000
```

### Environment Management

```bash
# Development
export ENV=development
export LOG_LEVEL=debug

# Staging
export ENV=staging
export LOG_LEVEL=info

# Production
export ENV=production
export LOG_LEVEL=warning
```

### Database Migrations

```bash
# Install Alembic
pip install alembic

# Initialize migrations
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add user_id to artifacts"

# Apply migrations
alembic upgrade head
```

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Refused

**Symptom**: Frontend shows "Disconnected"

**Solutions**:
- Ensure backend is running on port 8000
- Check CORS configuration
- Verify WebSocket URL in `app.js` matches backend

#### 2. GPT-5 API Errors

**Symptom**: "Error processing message" in terminal

**Solutions**:
- Verify `OPENAI_API_KEY` in `.env`
- Check API quota and billing
- Ensure model name is correct (`gpt-5`)

#### 3. Database Connection Failed

**Symptom**: "Connection refused" on startup

**Solutions**:
- Check PostgreSQL container: `docker ps`
- Verify port 5433 is not in use
- Check `DATABASE_URL` in `.env`

#### 4. Artifact Not Rendering

**Symptom**: Artifact created but shows JSON instead of rendered view

**Solutions**:
- Verify artifact type matches renderer name
- Check browser console for JavaScript errors
- Ensure external libraries (Chart.js, etc.) loaded

---

## API Reference

### WebSocket Events

| Event Type | Direction | Description |
|------------|-----------|-------------|
| `user_message` | Client → Server | User sends a message |
| `assistant_message` | Server → Client | Agent's text response |
| `reasoning` | Server → Client | Agent's reasoning process |
| `tool_execution` | Server → Client | Tool execution status |
| `artifact_created` | Server → Client | New artifact created |
| `error` | Server → Client | Error occurred |

### Database Models

#### Artifact

```python
class Artifact(Base):
    id: UUID              # Primary key
    type: str             # Artifact type
    status: ArtifactStatus # intermediate | final
    data: dict            # JSONB artifact payload
    artifact_metadata: dict # Optional metadata
    created_at: datetime  # Creation timestamp
```

### Pydantic Schemas

All artifact schemas in `backend/schemas.py`:

- `CsvArtifact`
- `ChartArtifact`
- `CodeArtifact`
- `MarkdownArtifact`
- `HtmlArtifact`
- `PaymentLinkArtifact`

---

## Roadmap

### Short Term (1-2 months)
- ✅ POC with 6 artifact types
- ⏳ Streaming responses
- ⏳ User authentication
- ⏳ Artifact versioning

### Medium Term (3-6 months)
- ⏳ Real-time collaboration
- ⏳ Advanced artifact types (diagrams, maps)
- ⏳ Mobile responsive UI
- ⏳ Artifact export/import

### Long Term (6-12 months)
- ⏳ Plugin system for custom artifacts
- ⏳ AI-powered artifact suggestions
- ⏳ Team workspaces
- ⏳ API for third-party integrations

---

## Contributing

### Adding a New Artifact Type

1. **Define the schema** in `backend/schemas.py`
2. **Add the tool** to `backend/agent.py` TOOLS list
3. **Update artifact_mapping** in `execute_artifact_tool()`
4. **Create renderer** in `frontend/app.js` ArtifactRenderers
5. **Test** with various inputs
6. **Document** in README.md

### Code Style

- **Python**: PEP 8, type hints, docstrings
- **JavaScript**: ESLint, camelCase
- **SQL**: Lowercase keywords, snake_case

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/ottic/ottic-agentic/issues
- Documentation: https://docs.ottic.com
- Email: support@ottic.com
