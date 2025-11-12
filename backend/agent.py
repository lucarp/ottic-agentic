"""GPT-5 agent with typed artifact tools."""

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

from database import create_artifact, ArtifactStatus

# Load environment variables
load_dotenv()
from schemas import (
    CsvArtifact,
    HtmlArtifact,
    ChartArtifact,
    PaymentLinkArtifact,
    MarkdownArtifact,
    CodeArtifact,
    FetchedLink,
)


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool definitions for GPT-5 Responses API
TOOLS = [
    # OpenAI native web search tool
    {
        "type": "web_search"
    },
    {
        "type": "function",
        "name": "create_csv_artifact",
        "description": (
            "Creates a CSV data artifact with headers and rows. "
            "Use this when you need to display tabular data, datasets, or spreadsheet-like information."
        ),
        "parameters": CsvArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "create_html_artifact",
        "description": (
            "Creates an HTML artifact that will be rendered in the browser. "
            "Use this for rich formatted content, interactive elements, or custom layouts."
        ),
        "parameters": HtmlArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "create_chart_artifact",
        "description": (
            "Creates a chart/visualization artifact using Chart.js. "
            "Supports bar, line, pie, doughnut, scatter, and other chart types. "
            "Perfect for data visualization and analytics."
        ),
        "parameters": ChartArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "create_payment_link_artifact",
        "description": (
            "Creates a Stripe payment link artifact. "
            "Use this when the user needs to make a payment, purchase, or donation."
        ),
        "parameters": PaymentLinkArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "create_markdown_artifact",
        "description": (
            "Creates a markdown document artifact. "
            "Use this for formatted text documents, reports, documentation, or written content."
        ),
        "parameters": MarkdownArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "create_code_artifact",
        "description": (
            "Creates a code snippet artifact with syntax highlighting. "
            "Use this to show code examples, scripts, or programming solutions."
        ),
        "parameters": CodeArtifact.model_json_schema()
    },
    {
        "type": "function",
        "name": "fetch_url_content",
        "description": (
            "Fetches and extracts clean content from a specific URL. "
            "Use this when you need to read, analyze, or extract information from a specific webpage. "
            "Returns the main content in markdown format, along with metadata like title and publish date. "
            "Best for when you need the full content of a page, not just search results."
        ),
        "parameters": FetchedLink.model_json_schema()
    },
]


async def fetch_url_content_impl(url: str) -> dict[str, Any]:
    """
    Fetch and extract content from a URL.
    Returns structured data with title, content (markdown), and metadata.
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {
                "error": "Invalid URL format",
                "url": url
            }

        # Fetch the page
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; OtticBot/1.0)"
            })
            response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = None
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else None
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()

        # Remove script, style, and navigation elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        # Try to find main content
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.main-content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content found, use body
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text content
        text_content = main_content.get_text(separator='\n', strip=True)

        # Clean up excessive newlines
        lines = [line.strip() for line in text_content.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        clean_content = '\n\n'.join(lines)

        # Extract metadata
        metadata = {}

        # Try to find author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            metadata['author'] = author_meta['content']

        # Try to find publish date
        date_meta = soup.find('meta', attrs={'property': 'article:published_time'}) or \
                    soup.find('meta', attrs={'name': 'date'})
        if date_meta and date_meta.get('content'):
            metadata['published_date'] = date_meta['content']

        # Try to find description
        desc_meta = soup.find('meta', attrs={'name': 'description'}) or \
                    soup.find('meta', attrs={'property': 'og:description'})
        if desc_meta and desc_meta.get('content'):
            metadata['description'] = desc_meta['content']

        return {
            "url": url,
            "title": title or "Untitled",
            "content": clean_content,
            "content_type": "text",
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata if metadata else None
        }

    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "url": url
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "url": url
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch content: {str(e)}",
            "url": url
        }


async def execute_artifact_tool(tool_name: str, tool_input: dict[str, Any], db: Session) -> dict[str, Any]:
    """Execute an artifact creation tool."""
    try:
        # Map tool name to artifact type and schema
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
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

        artifact_type, schema_class = artifact_mapping[tool_name]

        # Log the input for debugging
        sys.stderr.write(f"\n=== TOOL EXECUTION DEBUG ===\n")
        sys.stderr.write(f"Tool name: {tool_name}\n")
        sys.stderr.write(f"Tool input type: {type(tool_input)}\n")
        sys.stderr.write(f"Tool input: {tool_input}\n")
        sys.stderr.flush()

        # Parse JSON string if needed
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)
            sys.stderr.write(f"Parsed tool_input: {tool_input}\n")
            sys.stderr.flush()

        # Special handling for fetch_url_content - fetch the actual URL
        if tool_name == "fetch_url_content":
            url = tool_input.get("url")
            if not url:
                return {
                    "success": False,
                    "error": "URL is required for fetch_url_content"
                }

            # Fetch the URL content
            fetched_data = await fetch_url_content_impl(url)

            if "error" in fetched_data:
                return {
                    "success": False,
                    "error": fetched_data["error"]
                }

            # Use the fetched data as the tool_input
            tool_input = fetched_data

        # Validate input against Pydantic schema
        validated_data = schema_class(**tool_input)

        sys.stderr.write(f"Validated data: {validated_data}\n")
        sys.stderr.flush()

        # Create artifact in database
        artifact = create_artifact(
            db=db,
            artifact_type=artifact_type,
            data=validated_data.model_dump(),
            status=ArtifactStatus.FINAL,
            artifact_metadata={"tool": tool_name}
        )

        return {
            "success": True,
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "message": f"Successfully created {artifact_type} artifact"
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        sys.stderr.write(f"\n=== ERROR CREATING ARTIFACT ===\n")
        sys.stderr.write(f"{error_details}\n")
        sys.stderr.write(f"================================\n")
        sys.stderr.flush()
        return {
            "success": False,
            "error": f"Failed to create artifact: {str(e)}",
            "details": error_details
        }


async def run_agent(
    user_input: str,
    db: Session,
    previous_response_id: Optional[str] = None
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the GPT-5 agent with user input.

    Yields streaming events for the frontend to render.
    """

    sys.stderr.write(f"\n=== RUN_AGENT CALLED ===\n")
    sys.stderr.write(f"User input: {user_input}\n")
    sys.stderr.write(f"Previous response ID: {previous_response_id}\n")
    sys.stderr.flush()

    try:
        sys.stderr.write(f"Creating GPT-5 response...\n")
        sys.stderr.flush()

        # Create initial response with GPT-5 (STREAMING ENABLED)
        try:
            response_stream = await client.responses.create(
                model="gpt-5",
                input=user_input,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
                tools=TOOLS,
                previous_response_id=previous_response_id,
                stream=True  # ✅ STREAMING ENABLED
            )
            sys.stderr.write(f"GPT-5 streaming response created\n")
            sys.stderr.flush()
        except Exception as api_error:
            sys.stderr.write(f"\n=== GPT-5 API ERROR ===\n")
            sys.stderr.write(f"Error type: {type(api_error).__name__}\n")
            sys.stderr.write(f"Error message: {str(api_error)}\n")
            sys.stderr.flush()
            raise

        # Process streaming chunks
        response_id = None
        text_buffer = ""
        reasoning_buffer = ""
        tool_calls = {}  # Buffer for accumulating tool call arguments

        # Loop to handle streaming chunks
        async for chunk in response_stream:
            # Capture response ID from first chunk
            if response_id is None and hasattr(chunk, 'response_id'):
                response_id = chunk.response_id
                sys.stderr.write(f"Response ID: {response_id}\n")
                sys.stderr.flush()

            chunk_type = chunk.type if hasattr(chunk, 'type') else None

            # Handle text deltas (STREAMING)
            if chunk_type == "response.output_item.added":
                # New output item starting
                pass

            elif chunk_type == "response.output_item.done":
                # Output item completed - flush any buffers
                if text_buffer:
                    yield {
                        "type": "text_done",
                        "response_id": response_id
                    }
                    text_buffer = ""

            elif chunk_type == "response.content_part.added":
                # Content part added
                pass

            elif chunk_type == "response.content_part.delta":
                # STREAMING TEXT DELTA
                if hasattr(chunk, 'delta') and chunk.delta:
                    text_buffer += chunk.delta

                    # Send delta if buffer reaches threshold (simple chunking)
                    if len(text_buffer) >= 50:
                        yield {
                            "type": "text_delta",
                            "delta": text_buffer,
                            "response_id": response_id
                        }
                        text_buffer = ""

            elif chunk_type == "response.content_part.done":
                # Flush remaining text buffer
                if text_buffer:
                    yield {
                        "type": "text_delta",
                        "delta": text_buffer,
                        "response_id": response_id
                    }
                    text_buffer = ""

            # Handle reasoning (can be done or delta)
            elif chunk_type == "response.output_item.done" and hasattr(chunk, 'item'):
                if chunk.item.type == "message":
                    # Final message - already handled via deltas
                    pass

            # Legacy fallback for non-streaming chunks
            elif hasattr(chunk, 'type') and chunk.type == "reasoning":
                    sys.stderr.write(f"Found reasoning item\n")
                    sys.stderr.write(f"Has summary: {hasattr(item, 'summary')}\n")
                    if hasattr(item, 'summary'):
                        sys.stderr.write(f"Summary value: {item.summary}\n")
                    sys.stderr.flush()

                    # Reasoning items have a summary attribute which is an array
                    if hasattr(item, 'summary') and item.summary:
                        # Extract text from summary items
                        summary_texts = []
                        for summary_item in item.summary:
                            if hasattr(summary_item, 'text'):
                                # Ensure we convert to string
                                summary_texts.append(str(summary_item.text))

                        sys.stderr.write(f"Extracted {len(summary_texts)} summary texts\n")
                        sys.stderr.flush()

                        if summary_texts:
                            yield {
                                "type": "reasoning",
                                "content": " ".join(summary_texts),
                                "response_id": response_id
                            }

                # Handle text/message output
                elif item.type == "text" or item.type == "message":
                    sys.stderr.write(f"Found text/message item, extracting content\n")
                    sys.stderr.write(f"Content type: {type(item.content)}\n")
                    sys.stderr.flush()

                    # Extract text content - could be a string or an object with text attribute
                    text_content = ""
                    if isinstance(item.content, str):
                        text_content = item.content
                    elif hasattr(item.content, 'text'):
                        # Ensure we convert to string
                        text_content = str(item.content.text) if item.content.text else ""
                    elif isinstance(item.content, list):
                        # Content might be an array of text items
                        text_parts = []
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                text_parts.append(str(content_item.text))
                            elif isinstance(content_item, str):
                                text_parts.append(content_item)
                        text_content = " ".join(text_parts)
                    else:
                        # Fallback: try to convert to string
                        text_content = str(item.content)

                    sys.stderr.write(f"Extracted text type: {type(text_content)}\n")
                    sys.stderr.write(f"Extracted text: {text_content[:100] if text_content else '(empty)'}\n")
                    sys.stderr.flush()

                    if text_content:
                        yield {
                            "type": "assistant_message",
                            "content": text_content,
                            "response_id": response_id
                        }

                # Handle tool calls
                elif item.type == "function_call":
                    has_function_calls = True
                    tool_name = item.name
                    tool_input = item.arguments
                    tool_call_id = item.call_id

                    # Notify frontend that tool is executing
                    yield {
                        "type": "tool_execution",
                        "tool_name": tool_name,
                        "status": "started",
                        "input": tool_input if isinstance(tool_input, dict) else None
                    }

                    # Execute the tool
                    result = await execute_artifact_tool(tool_name, tool_input, db)

                    # Collect tool output for submission back to GPT-5
                    output_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output_str
                    })

                    # Notify completion
                    yield {
                        "type": "tool_execution",
                        "tool_name": tool_name,
                        "status": "completed" if result.get("success") else "failed",
                        "output": result
                    }

                    # If artifact was created successfully, send artifact_created event
                    if result.get("success"):
                        from database import get_artifact_by_id
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

            # If there were no function calls, we're done
            if not has_function_calls:
                break

            # Submit tool outputs and get the next response
            response = await client.responses.create(
                model="gpt-5",
                input=tool_outputs,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
                tools=TOOLS,
                previous_response_id=response_id,
                stream=False
            )

            response_id = response.id

        # Store response_id for next turn
        yield {
            "type": "response_complete",
            "response_id": response_id
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        sys.stderr.write(f"\n=== AGENT ERROR ===\n")
        sys.stderr.write(f"{error_details}\n")
        sys.stderr.write(f"===================\n")
        sys.stderr.flush()
        yield {
            "type": "error",
            "error": str(e)
        }
