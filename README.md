# Agentic Artifact System POC

A proof-of-concept demonstrating an AI agent that can create and manage typed artifacts using GPT-5 and the Responses API.

## Architecture

- **Backend**: FastAPI + WebSocket + PostgreSQL
- **Frontend**: Vanilla JS with specialized renderers for each artifact type
- **AI Model**: GPT-5 with medium reasoning effort
- **Artifacts**: Each artifact type is a separate tool with Pydantic validation

## Supported Artifact Types

Each artifact type has:
- A Pydantic schema for validation
- A dedicated tool for the agent
- A specialized frontend renderer

| Type | Description | Use Case |
|------|-------------|----------|
| `csv` | Tabular data with headers and rows | Datasets, spreadsheets |
| `html` | Custom HTML content | Rich formatting, interactive elements |
| `chart` | Chart.js visualizations | Data visualization, analytics |
| `payment_link` | Stripe payment interface | Payments, donations |
| `markdown` | Markdown documents | Reports, documentation |
| `code` | Syntax-highlighted code snippets | Code examples, scripts |
| `fetched_link` | Extracted web page content | Reading articles, documentation |
| `domain_overview` | SEO domain performance metrics | Traffic analysis, keyword tracking |
| `competitor_analysis` | SEO competitor identification | Competitive research, market analysis |
| `keyword_research` | Keyword opportunities and gaps | Content strategy, SEO optimization |

## Web Search & URL Fetching

The platform supports two ways to access web content:

### 1. Web Search (OpenAI Native)
Uses OpenAI's built-in `web_search` tool for real-time internet access with:
- Automatic citation extraction
- Source attribution
- Up-to-date information retrieval

### 2. URL Content Fetching
Custom tool to fetch and extract content from specific URLs:
- Clean content extraction (removes ads, navigation)
- Automatic metadata parsing (title, author, date)
- Markdown-formatted output
- Perfect for reading full articles or documentation

## SEO Analysis (SE Ranking API)

The platform integrates with **SE Ranking** to provide professional SEO analysis capabilities:

### Domain SEO Overview
Analyze any domain's search performance:
- Organic & paid traffic estimates
- Total keywords ranking
- Traffic value in multiple currencies
- Monthly visitor projections

### Competitor Analysis
Identify and analyze SEO competitors:
- Find domains competing for the same keywords
- See keyword overlap counts
- Supports regional databases (US, UK, CA, DE, etc.)
- Organic and paid search analysis

### Keyword Research
Discover keyword opportunities:
- **Similar Keywords**: Find semantically related terms to expand your content
- **Keyword Gap Analysis**: Identify keywords competitors rank for that you don't
- Search volume, CPC, and difficulty scores
- Color-coded difficulty ratings (Easy/Medium/Hard)

**Powered by:** SE Ranking Data API

## Setup

### 1. Start PostgreSQL

```bash
cd poc
docker-compose up -d
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - SERANKING_API_KEY (required for SEO features)
```

**Note**: Get your SE Ranking API key from [SE Ranking API Dashboard](https://seranking.com/api)

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run Backend

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### 5. Open Frontend

```bash
cd frontend
# Serve with any HTTP server, e.g.:
python -m http.server 3000
```

Then open `http://localhost:3000` in your browser.

## Usage Examples

### Create a CSV Artifact
```
Show me the top 5 programming languages in a table
```

### Create a Chart
```
Create a bar chart showing sales data: Q1: 1000, Q2: 1500, Q3: 1200, Q4: 1800
```

### Create a Payment Link
```
Create a payment link for $49.99 for a premium subscription
```

### Create a Code Snippet
```
Show me a Python function to calculate factorial
```

### Create a Markdown Document
```
Write a project summary document about this POC
```

### Search the Web
```
What are the latest developments in AI reasoning models?
```

### Fetch and Read a URL
```
Can you read and summarize this article: https://example.com/article
```

### SEO Domain Analysis
```
Analyze the SEO performance of stripe.com
```

### Find SEO Competitors
```
Who are the top 10 competitors for shopify.com in the US?
```

### Keyword Research - Similar Keywords
```
Find keywords similar to "payment gateway"
```

### Keyword Gap Analysis
```
What keywords does paypal.com rank for that stripe.com doesn't?
```

## How It Works

1. **User sends message** via WebSocket
2. **GPT-5 agent processes** with medium reasoning effort
3. **Agent decides which artifact tool to call** based on context
4. **Tool executes** and validates input with Pydantic
5. **Artifact saved to PostgreSQL** with full data
6. **Frontend receives artifact** and routes to specialized renderer
7. **Renderer displays** artifact in appropriate format

## Key Features

✅ **Typed Artifacts**: Each artifact type has strict schema validation
✅ **Specialized Renderers**: Frontend knows exactly how to display each type
✅ **Real-time Updates**: WebSocket streams messages and artifacts
✅ **Persistent Storage**: PostgreSQL stores all artifacts
✅ **Extensible**: Easy to add new artifact types

## Adding New Artifact Types

1. **Define schema** in `backend/schemas.py`:
```python
class NewArtifact(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(description="...")
```

2. **Add tool** in `backend/agent.py`:
```python
{
    "type": "function",
    "function": {
        "name": "create_new_artifact",
        "description": "...",
        "parameters": NewArtifact.model_json_schema()
    }
}
```

3. **Add renderer** in `frontend/app.js`:
```javascript
ArtifactRenderers.new_type = (data) => {
    return `<div>Custom rendering for ${data.field1}</div>`;
};
```

## Technology Stack

### Backend
- FastAPI (ASGI web framework)
- OpenAI Python SDK (GPT-5 Responses API)
- SQLAlchemy (PostgreSQL ORM)
- Pydantic (schema validation)
- WebSockets (real-time communication)

### Frontend
- Vanilla JavaScript
- Chart.js (visualizations)
- Marked.js (markdown rendering)
- Highlight.js (code syntax highlighting)

## API Endpoints

- `GET /` - Health check
- `GET /health` - Health status
- `WS /ws` - WebSocket connection for agent interaction

## WebSocket Message Protocol

### Client → Server
```json
{
  "type": "user_message",
  "content": "Your message here"
}
```

### Server → Client
```json
// Assistant text
{"type": "assistant_message", "content": "..."}

// Tool execution
{"type": "tool_execution", "tool_name": "...", "status": "started|completed|failed"}

// Artifact created
{"type": "artifact_created", "artifact": {...}}

// Error
{"type": "error", "error": "..."}
```

## Database Schema

```sql
CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,  -- 'intermediate' or 'final'
    data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL
);
```

## Future Enhancements

- [ ] Streaming responses from GPT-5
- [ ] Artifact versioning and updates
- [ ] Multi-user support with auth
- [ ] Artifact export (PDF, CSV download)
- [ ] Collaborative editing
- [ ] Artifact templates
- [ ] Advanced chart types (D3.js)
- [ ] Real Stripe integration
- [ ] Intermediate artifact updates

## License

MIT
