# Quick Start Guide

## What We Built

✅ **Complete POC with:**
- GPT-5 agent with Responses API (medium reasoning)
- 6 typed artifact tools (CSV, HTML, Chart, Payment Link, Markdown, Code)
- Each artifact has Pydantic schema validation
- Specialized frontend renderers for each type
- PostgreSQL persistence
- Real-time WebSocket communication
- Split-screen terminal + artifact viewer

## File Structure

```
poc/
├── docker-compose.yml          # PostgreSQL container
├── .env.example                # Environment template
├── README.md                   # Full documentation
├── QUICKSTART.md               # This file
├── backend/
│   ├── main.py                 # FastAPI + WebSocket server
│   ├── agent.py                # GPT-5 agent with 6 artifact tools
│   ├── database.py             # PostgreSQL models
│   ├── schemas.py              # Pydantic artifact schemas
│   └── requirements.txt        # Python dependencies
└── frontend/
    ├── index.html              # Split-screen UI
    └── app.js                  # WebSocket client + renderers
```

## Running the POC

### 1. Set up PostgreSQL

```bash
cd poc
docker-compose up -d
# Wait for container to be healthy (check with: docker ps)
```

### 2. Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run Backend Server

```bash
# From poc/backend directory
python main.py
```

Server will start at `http://localhost:8000`

### 5. Open Frontend

Open `poc/frontend/index.html` directly in your browser, or serve it:

```bash
cd poc/frontend
python -m http.server 3000
```

Then navigate to `http://localhost:3000`

## Test the POC

Try these prompts:

### CSV Artifact
```
Create a table showing the top 5 programming languages with their popularity scores
```

### Chart Artifact
```
Create a bar chart showing quarterly sales: Q1: $45k, Q2: $62k, Q3: $58k, Q4: $71k
```

### Code Artifact
```
Show me a Python function to calculate fibonacci numbers
```

### Markdown Artifact
```
Write a project summary document about this POC
```

### HTML Artifact
```
Create a colorful welcome card with my name
```

### Payment Link Artifact
```
Create a payment link for $29.99 for a premium subscription
```

## Architecture Highlights

### Typed Artifacts (backend/schemas.py)
Each artifact type has a Pydantic model:
- `CsvArtifact`: headers + rows
- `ChartArtifact`: chart_type + labels + datasets (Chart.js format)
- `CodeArtifact`: code + language + title
- `MarkdownArtifact`: content + title
- `HtmlArtifact`: html + css + title
- `PaymentLinkArtifact`: amount + currency + description

### Dedicated Tools (backend/agent.py)
Each artifact type is a separate tool:
- `create_csv_artifact`
- `create_chart_artifact`
- `create_code_artifact`
- `create_markdown_artifact`
- `create_html_artifact`
- `create_payment_link_artifact`

### Specialized Renderers (frontend/app.js)
Frontend knows how to render each type:
- **CSV**: Styled HTML table
- **Chart**: Chart.js canvas
- **Code**: Highlight.js syntax highlighting
- **Markdown**: Marked.js rendering
- **HTML**: Sandboxed iframe rendering
- **Payment Link**: Stripe-like UI (mock)

## What Makes This Different

1. **No Generic JSON Blobs**: Each artifact has a strict schema
2. **Tool per Type**: GPT-5 calls specific tools, not generic "create artifact"
3. **Renderer per Type**: Frontend has specialized rendering logic
4. **Extensible**: Add new artifact types by:
   - Adding Pydantic schema in `schemas.py`
   - Adding tool in `agent.py`
   - Adding renderer in `app.js`

## Next Steps

- Try different artifact types
- Add new artifact types (e.g., `TableArtifact`, `DiagramArtifact`)
- Implement streaming responses
- Add artifact export functionality
- Integrate real Stripe payments

## Troubleshooting

**PostgreSQL not starting?**
```bash
docker-compose logs postgres
```

**Backend errors?**
- Check `.env` has correct `OPENAI_API_KEY`
- Verify PostgreSQL is running: `docker ps`
- Check logs in terminal

**Frontend not connecting?**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify WebSocket URL in `app.js` matches backend

## Technology Stack

| Layer | Technology |
|-------|------------|
| AI Model | GPT-5 (Responses API) |
| Backend | FastAPI + SQLAlchemy |
| Database | PostgreSQL 16 |
| Real-time | WebSockets |
| Frontend | Vanilla JS |
| Validation | Pydantic |
| Charts | Chart.js |
| Markdown | Marked.js |
| Code Highlighting | Highlight.js |
