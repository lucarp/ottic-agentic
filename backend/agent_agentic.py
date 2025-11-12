"""Multi-turn agentic implementation using OpenAI Responses API properly."""

import asyncio
import json
import logging
import os
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

logger = logging.getLogger(__name__)

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


async def run_agent_agentic(
    user_input: str,
    db: Session,
    conversation_history: list[dict[str, str]] = None
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run GPT-5 agent with PROPER MULTI-TURN AGENTIC PATTERN.

    This implements the correct Responses API pattern:
    1. Create response (agent plans)
    2. Execute tool calls locally
    3. Submit tool outputs back to agent
    4. Agent continues with results
    5. Repeat until complete
    """
    try:
        logger.info(f"🤖 AGENTIC MODE: Starting multi-turn agent for: '{user_input[:50]}...'")

        # Format input with conversation context
        formatted_input = user_input
        if conversation_history and len(conversation_history) > 0:
            logger.info(f"📚 Using conversation history ({len(conversation_history)} messages)")
            context_parts = ["Previous conversation:"]
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")

            context_parts.append(f"\nCurrent question: {user_input}")
            formatted_input = "\n".join(context_parts)

        # TURN 1: Create initial response (agent makes plan)
        logger.info("🔄 TURN 1: Creating initial response...")

        # Notify frontend that agent is starting
        yield {"type": "text_delta", "delta": "🤖 Starting multi-turn agent...\n"}

        # Add timeout to initial create call
        try:
            async with asyncio.timeout(120.0):  # 2-minute timeout
                response = await client.responses.create(
                    model="gpt-5",
                    input=formatted_input,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "medium"},
                    tools=TOOLS,
                )
        except asyncio.TimeoutError:
            logger.error("⏱️  TIMEOUT: Initial response.create() exceeded 120s")
            yield {
                "type": "error",
                "error": "Request timeout: The AI model took longer than 120s to create the initial response. This can happen with very complex requests. Please try simplifying your request or try again."
            }
            return

        response_id = response.id
        logger.info(f"✅ Response created: {response_id}")

        # Multi-turn loop
        max_turns = 10
        turn = 1

        while turn <= max_turns:
            logger.info(f"🔄 TURN {turn}: Processing response status={response.status}")

            # Notify frontend of current turn
            yield {"type": "text_delta", "delta": f"📍 Turn {turn}: Agent analyzing...\n"}

            # Log all outputs for debugging
            logger.info(f"📋 Response has {len(response.output)} outputs")
            for idx, output in enumerate(response.output):
                logger.info(f"  Output {idx}: type={output.type}")

            # Check if there are function calls to execute (even if status is completed)
            has_function_calls = any(output.type == "function_call" for output in response.output)

            if response.status == "completed" and not has_function_calls:
                logger.info(f"✅ Agent completed after {turn} turns (no function calls)")

                # Send final text output if any
                for output in response.output:
                    if output.type == "text":
                        yield {"type": "text_delta", "delta": output.content, "response_id": response_id}

                yield {"type": "response_complete", "response_id": response_id}
                break

            # If completed but has function calls, execute them
            if response.status == "completed" and has_function_calls:
                logger.info(f"✅ Agent completed with function calls - executing tools...")
                # Fall through to tool execution below

            # If requires action OR completed with function calls, execute tools
            if response.status == "requires_action" or (response.status == "completed" and has_function_calls):
                logger.info(f"🔧 Agent requires action - executing tools...")

                tool_calls = []
                tool_outputs = []

                # Collect all tool calls
                for output in response.output:
                    if output.type == "function_call":
                        tool_calls.append(output)
                        logger.info(f"🔧 Tool call: {output.name}")

                # Notify frontend of tool execution phase
                if tool_calls:
                    yield {"type": "text_delta", "delta": f"🔧 Executing {len(tool_calls)} tool(s)...\n"}

                # Execute each tool
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments

                    # Notify frontend
                    yield {"type": "tool_execution", "tool_name": tool_name, "status": "started"}

                    # Handle web_search (native OpenAI tool - no local execution)
                    if tool_name == "web_search":
                        logger.info(f"🌐 Web search will be handled by OpenAI")
                        # Web search is handled by OpenAI - we'll get results in next turn
                        continue

                    # Execute artifact tools locally
                    if tool_name in ['create_csv_artifact', 'create_html_artifact', 'create_chart_artifact',
                                       'create_payment_link_artifact', 'create_markdown_artifact',
                                       'create_code_artifact', 'fetch_url_content']:
                        result = await execute_artifact_tool(tool_name, tool_args, db)

                        # Notify frontend
                        yield {"type": "tool_execution", "tool_name": tool_name,
                               "status": "completed" if result.get("success") else "failed",
                               "output": result}

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

                        # Prepare tool output for submission
                        tool_outputs.append({
                            "call_id": tool_call.call_id,
                            "output": json.dumps(result)
                        })

                # If status was "completed", we're done after executing tools
                if response.status == "completed":
                    logger.info("✅ Tools executed, agent was already completed - finishing")
                    yield {"type": "response_complete", "response_id": response_id}
                    break

                # Otherwise submit tool outputs and continue
                if tool_outputs:
                    logger.info(f"📤 Submitting {len(tool_outputs)} tool outputs back to agent...")
                    yield {"type": "text_delta", "delta": f"📤 Submitting results to agent for turn {turn + 1}...\n"}
                    response = await client.responses.submit_tool_outputs(
                        response_id=response_id,
                        tool_outputs=tool_outputs
                    )
                    turn += 1
                else:
                    # No local tools executed, just continue
                    logger.info("⏭️  No local tools to submit, fetching updated response...")
                    yield {"type": "text_delta", "delta": "⏭️  Fetching updated response...\n"}
                    response = await client.responses.retrieve(response_id)
                    await asyncio.sleep(1)  # Wait a bit for OpenAI to process
                    turn += 1

            else:
                logger.warning(f"⚠️  Unexpected response status: {response.status}")
                break

        if turn > max_turns:
            logger.error(f"⏱️  Max turns ({max_turns}) exceeded")
            yield {"type": "error", "error": f"Agent exceeded maximum turns ({max_turns})"}

    except Exception as e:
        logger.error(f"❌ ERROR in agentic agent: {type(e).__name__}: {e}", exc_info=True)
        yield {"type": "error", "error": str(e)}
