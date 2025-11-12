"""Simple production-ready streaming implementation for GPT-5 agent."""

import json
import os
import sys
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from database import create_artifact, ArtifactStatus, get_artifact_by_id
from schemas import (
    CsvArtifact,
    HtmlArtifact,
    ChartArtifact,
    PaymentLinkArtifact,
    MarkdownArtifact,
    CodeArtifact,
    FetchedLink,
)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool definitions
TOOLS = [
    {"type": "web_search"},  # OpenAI's native web search tool
    {"type": "function", "name": "create_csv_artifact", "description": "Creates a CSV data artifact with headers and rows.", "parameters": CsvArtifact.model_json_schema()},
    {"type": "function", "name": "create_html_artifact", "description": "Creates an HTML artifact that will be rendered in the browser.", "parameters": HtmlArtifact.model_json_schema()},
    {"type": "function", "name": "create_chart_artifact", "description": "Creates a chart/visualization artifact using Chart.js.", "parameters": ChartArtifact.model_json_schema()},
    {"type": "function", "name": "create_payment_link_artifact", "description": "Creates a Stripe payment link artifact.", "parameters": PaymentLinkArtifact.model_json_schema()},
    {"type": "function", "name": "create_markdown_artifact", "description": "Creates a markdown document artifact.", "parameters": MarkdownArtifact.model_json_schema()},
    {"type": "function", "name": "create_code_artifact", "description": "Creates a code snippet artifact with syntax highlighting.", "parameters": CodeArtifact.model_json_schema()},
    {"type": "function", "name": "fetch_url_content", "description": "Fetches and extracts clean content from a specific URL.", "parameters": FetchedLink.model_json_schema()},
]


async def fetch_url_content_impl(url: str) -> dict[str, Any]:
    """Fetch and extract content from URL."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"error": "Invalid URL format", "url": url}

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as http_client:
            response = await http_client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; OtticBot/1.0)"})
            response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()

        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.main-content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body if soup.body else soup

        text_content = main_content.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        clean_content = '\n\n'.join(lines)

        metadata = {}
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            metadata['author'] = author_meta['content']

        date_meta = soup.find('meta', attrs={'property': 'article:published_time'}) or soup.find('meta', attrs={'name': 'date'})
        if date_meta and date_meta.get('content'):
            metadata['published_date'] = date_meta['content']

        return {
            "url": url,
            "title": title or "Untitled",
            "content": clean_content,
            "content_type": "text",
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata if metadata else None
        }

    except Exception as e:
        return {"error": f"Failed to fetch: {str(e)}", "url": url}


async def execute_artifact_tool(tool_name: str, tool_input: dict[str, Any], db: Session) -> dict[str, Any]:
    """Execute artifact tool."""
    try:
        artifact_mapping = {
            "create_csv_artifact": ("csv", CsvArtifact),
            "create_html_artifact": ("html", HtmlArtifact),
            "create_chart_artifact": ("chart", ChartArtifact),
            "create_payment_link_artifact": ("payment_link", PaymentLinkArtifact),
            "create_markdown_artifact": ("markdown", MarkdownArtifact),
            "create_code_artifact": ("code", CodeArtifact),
            "fetch_url_content": ("fetched_link", FetchedLink),
        }

        if tool_name not in artifact_mapping:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        artifact_type, schema_class = artifact_mapping[tool_name]

        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)

        # Special handling for URL fetching
        if tool_name == "fetch_url_content":
            url = tool_input.get("url")
            if not url:
                return {"success": False, "error": "URL is required"}
            fetched_data = await fetch_url_content_impl(url)
            if "error" in fetched_data:
                return {"success": False, "error": fetched_data["error"]}
            tool_input = fetched_data

        validated_data = schema_class(**tool_input)
        artifact = create_artifact(db=db, artifact_type=artifact_type, data=validated_data.model_dump(), status=ArtifactStatus.FINAL, artifact_metadata={"tool": tool_name})

        return {
            "success": True,
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "message": f"Successfully created {artifact_type} artifact"
        }

    except Exception as e:
        return {"success": False, "error": f"Failed: {str(e)}"}


async def run_agent_streaming(
    user_input: str,
    db: Session,
    previous_response_id: Optional[str] = None
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run GPT-5 agent with STREAMING enabled.
    Simple, production-ready implementation.
    """
    try:
        # Create streaming response
        stream = await client.responses.create(
            model="gpt-5",
            input=user_input,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"},
            tools=TOOLS,
            previous_response_id=previous_response_id,
            stream=True
        )

        # Streaming state
        response_id = None
        text_buffer = ""
        current_output = None
        event_count = 0
        function_calls = {}  # Track function call metadata by item_id

        # Process stream
        async for event in stream:
            event_count += 1

            # Track response ID from event.response.id
            if not response_id:
                if hasattr(event, 'response') and hasattr(event.response, 'id') and event.response.id:
                    response_id = event.response.id

            event_type = event.type if hasattr(event, 'type') else None

            # Text delta streaming
            if event_type == "response.output_text.delta":
                if hasattr(event, 'delta'):
                    delta_text = str(event.delta)
                    text_buffer += delta_text
                    yield {"type": "text_delta", "delta": delta_text, "response_id": response_id}

            elif event_type == "response.text.done":
                # Text content is complete (no buffering needed since we send immediately)
                pass

            # Function call handling
            elif event_type == "response.function_call_arguments.delta":
                pass  # Tool arguments streaming

            elif event_type == "response.function_call_arguments.done":
                # Look up function call metadata by item_id
                if hasattr(event, 'item_id') and event.item_id in function_calls and hasattr(event, 'arguments'):
                    func_metadata = function_calls[event.item_id]
                    tool_name = func_metadata['name']
                    tool_args = event.arguments
                    call_id = func_metadata['call_id']

                    # Handle artifact creation tools
                    if tool_name in ['create_csv_artifact', 'create_html_artifact', 'create_chart_artifact',
                                       'create_payment_link_artifact', 'create_markdown_artifact',
                                       'create_code_artifact', 'fetch_url_content']:
                        # Notify start
                        yield {"type": "tool_execution", "tool_name": tool_name, "status": "started"}

                        # Execute
                        result = await execute_artifact_tool(tool_name, tool_args, db)

                        # Notify completion
                        yield {"type": "tool_execution", "tool_name": tool_name, "status": "completed" if result.get("success") else "failed", "output": result}

                        # Send artifact if created
                        if result.get("success"):
                            import uuid
                            artifact = get_artifact_by_id(db, uuid.UUID(result["artifact_id"]))
                            if artifact:
                                yield {
                                    "type": "artifact_created",
                                    "artifact": {
                                        "id": str(artifact.id),
                                        "type": artifact.type,
                                        "status": artifact.status.value,
                                        "data": artifact.data,
                                        "artifact_metadata": artifact.artifact_metadata,
                                        "created_at": artifact.created_at.isoformat()
                                    }
                                }

            elif event_type == "response.completed":
                # Final completion
                yield {"type": "response_complete", "response_id": response_id}
                break

            # Track output items (may contain function call names)
            elif event_type == "response.output_item.added":
                if hasattr(event, 'item') and hasattr(event.item, 'type') and event.item.type == 'function_call':
                    # Store function call metadata for later use
                    function_calls[event.item.id] = {
                        'name': event.item.name,
                        'call_id': event.item.call_id
                    }

    except Exception as e:
        print(f"ERROR in streaming agent: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        yield {"type": "error", "error": str(e)}
