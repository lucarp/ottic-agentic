"""Multi-turn agentic implementation using OpenAI Agents SDK."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Annotated, Any, AsyncGenerator
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, WebSearchTool
from sqlalchemy.orm import Session

from database import create_artifact, ArtifactStatus, get_artifact_by_id
from schemas import (
    CsvArtifact,
    HtmlArtifact,
    ChartArtifact,
    PaymentLinkArtifact,
    MarkdownArtifact,
    CodeArtifact,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# URL Fetching Tool
# ============================================================================

async def fetch_url_content_impl(url: str) -> dict[str, Any]:
    """Fetch and extract clean content from a URL."""
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"error": f"Invalid URL: {url}"}

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
                    tag.decompose()

                # Extract title
                title = soup.find("title")
                title_text = title.get_text().strip() if title else parsed_url.netloc

                # Extract main content
                main_content = soup.find("main") or soup.find("article") or soup.find("body")
                if main_content:
                    text_content = main_content.get_text(separator="\n", strip=True)
                else:
                    text_content = soup.get_text(separator="\n", strip=True)

                # Clean up excessive whitespace
                lines = [line.strip() for line in text_content.split("\n") if line.strip()]
                clean_content = "\n".join(lines)

                return {
                    "success": True,
                    "url": url,
                    "title": title_text,
                    "content": clean_content[:10000],  # Limit content size
                    "content_type": "text",
                    "fetch_timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # Non-HTML content
                return {
                    "success": True,
                    "url": url,
                    "title": parsed_url.netloc,
                    "content": response.text[:5000],
                    "content_type": content_type,
                    "fetch_timestamp": datetime.utcnow().isoformat(),
                }

    except Exception as e:
        logger.error(f"❌ URL fetch failed for {url}: {e}")
        return {"success": False, "error": str(e), "url": url}


@function_tool
async def fetch_url_content(
    url: Annotated[str, "The URL to fetch content from"]
) -> str:
    """Fetch and extract clean text content from a webpage or URL."""
    result = await fetch_url_content_impl(url)
    if result.get("success"):
        return f"Title: {result['title']}\n\nContent:\n{result['content']}"
    else:
        return f"Error fetching {url}: {result.get('error', 'Unknown error')}"


# ============================================================================
# Artifact Creation Tools Factory
# ============================================================================

def create_artifact_tools(db_session: Session):
    """Create artifact tools with injected database session."""

    @function_tool
    def create_csv_artifact(
        headers: Annotated[list[str], "Column headers for the CSV"],
        rows: Annotated[list[list], "Data rows as list of lists"],
        title: Annotated[str | None, "Optional title for the CSV"] = None,
        description: Annotated[str | None, "Optional description"] = None,
    ) -> str:
        """Create a CSV data artifact with headers and rows."""
        try:
            artifact_data = CsvArtifact(
                headers=headers,
                rows=rows,
                title=title,
                description=description
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="csv",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "csv",
                "message": "Successfully created CSV artifact"
            })
        except Exception as e:
            logger.error(f"Error creating CSV artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})


    @function_tool
    def create_chart_artifact(
        chart_type: Annotated[str, "Type of chart: 'bar', 'line', 'pie', 'doughnut', 'scatter'"],
        labels: Annotated[list[str], "Labels for the chart data points"],
        datasets: Annotated[list[dict], "Chart.js datasets with label, data, and styling"],
        title: Annotated[str | None, "Chart title"] = None,
        x_axis_label: Annotated[str | None, "X-axis label"] = None,
        y_axis_label: Annotated[str | None, "Y-axis label"] = None,
    ) -> str:
        """Create a chart/visualization artifact using Chart.js format."""
        try:
            artifact_data = ChartArtifact(
                chart_type=chart_type,
                labels=labels,
                datasets=datasets,
                title=title,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="chart",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "chart",
                "message": "Successfully created chart artifact"
            })
        except Exception as e:
            logger.error(f"Error creating chart artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    def create_markdown_artifact(
        content: Annotated[str, "Markdown formatted content"],
        title: Annotated[str | None, "Document title"] = None,
    ) -> str:
        """Create a markdown document artifact."""
        try:
            artifact_data = MarkdownArtifact(
                content=content,
                title=title
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="markdown",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "markdown",
                "message": "Successfully created markdown artifact"
            })
        except Exception as e:
            logger.error(f"Error creating markdown artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    def create_code_artifact(
        code: Annotated[str, "The code content"],
        language: Annotated[str, "Programming language (e.g., 'python', 'javascript', 'sql')"],
        title: Annotated[str | None, "Code snippet title"] = None,
        description: Annotated[str | None, "Description of what the code does"] = None,
    ) -> str:
        """Create a code snippet artifact with syntax highlighting."""
        try:
            artifact_data = CodeArtifact(
                code=code,
                language=language,
                title=title,
                description=description
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="code",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "code",
                "message": "Successfully created code artifact"
            })
        except Exception as e:
            logger.error(f"Error creating code artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    def create_html_artifact(
        html: Annotated[str, "The HTML content to render"],
        title: Annotated[str | None, "Optional title"] = None,
        css: Annotated[str | None, "Optional custom CSS styles"] = None,
    ) -> str:
        """Create an HTML content artifact."""
        try:
            artifact_data = HtmlArtifact(
                html=html,
                title=title,
                css=css
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="html",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "html",
                "message": "Successfully created HTML artifact"
            })
        except Exception as e:
            logger.error(f"Error creating HTML artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    def create_payment_link_artifact(
        amount: Annotated[float, "Payment amount in USD (must be greater than 0)"],
        description: Annotated[str, "Payment description"],
        currency: Annotated[str, "Currency code (e.g., 'usd', 'eur')"] = "usd",
        success_message: Annotated[str | None, "Message shown on success"] = None,
        metadata: Annotated[dict[str, str] | None, "Additional metadata"] = None,
    ) -> str:
        """Create a Stripe payment link artifact."""
        try:
            artifact_data = PaymentLinkArtifact(
                amount=amount,
                currency=currency,
                description=description,
                success_message=success_message,
                metadata=metadata
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="payment_link",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "payment_link",
                "message": "Successfully created payment link artifact"
            })
        except Exception as e:
            logger.error(f"Error creating payment link artifact: {e}")
            return json.dumps({"success": False, "error": str(e)})

    # Return all tools
    return [
        create_csv_artifact,
        create_chart_artifact,
        create_markdown_artifact,
        create_code_artifact,
        create_html_artifact,
        create_payment_link_artifact,
    ]


# ============================================================================
# Agent Setup with Tools
# ============================================================================

def create_agent_with_tools(db_session: Session) -> Agent:
    """Create an Agent with all artifact creation tools and web capabilities."""

    # Create artifact tools with injected database session
    artifact_tools = create_artifact_tools(db_session)

    # Create tool list
    tools = [
        WebSearchTool(),  # Built-in web search
        fetch_url_content,  # Custom URL fetching
    ] + artifact_tools  # All artifact creation tools

    agent = Agent(
        name="Artifact Assistant",
        instructions="""You are an AI assistant specialized in creating structured artifacts.

You have access to:
- Web search to find current information
- URL fetching to read content from webpages
- CSV creation for tabular data
- Chart creation for data visualizations
- Markdown documents for reports
- Code snippets with syntax highlighting
- HTML content for rich displays
- Payment links for transactions

When users request information or data:
1. Use web search if you need current information
2. Fetch URLs if you need to read specific webpages
3. Create appropriate artifacts to present the information clearly
4. Always create artifacts in the user's requested language
5. Provide clear, well-structured outputs

Be helpful, accurate, and create high-quality artifacts.""",
        tools=tools,
    )

    return agent


# ============================================================================
# Streaming Runner
# ============================================================================

async def run_agent_agentic(
    user_input: str,
    db: Session,
    conversation_history: list[dict[str, str]] | None = None
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agent with streaming support for the frontend.

    Yields events compatible with the existing WebSocket frontend.
    """
    logger.info(f"🤖 Starting Agent run for: '{user_input[:50]}...'")

    try:
        # Create agent with database session
        agent = create_agent_with_tools(db)

        # Build input with conversation history if provided
        full_input = user_input
        if conversation_history:
            # Format conversation history
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages for context
            ])
            full_input = f"Previous conversation:\n{history_text}\n\nCurrent request: {user_input}"

        # Run agent with streaming
        result = Runner.run_streamed(agent, full_input)

        current_text_delta = ""
        function_calls_active = {}

        async for event in result.stream_events():
            event_type = event.type

            # Handle text deltas (assistant messages)
            if event_type == "raw_response_event":
                if event.data.type == "response.text.delta":
                    delta = getattr(event.data, "delta", "")
                    current_text_delta += delta
                    yield {
                        "type": "text_delta",
                        "delta": delta
                    }

                elif event.data.type == "response.text.done":
                    if current_text_delta:
                        yield {
                            "type": "assistant_message",
                            "content": current_text_delta
                        }
                        current_text_delta = ""

                # Function call started
                elif event.data.type == "response.output_item.added":
                    if getattr(event.data.item, "type", None) == "function_call":
                        function_name = getattr(event.data.item, "name", "unknown")
                        call_id = getattr(event.data.item, "call_id", "unknown")
                        function_calls_active[call_id] = function_name

                        yield {
                            "type": "tool_execution",
                            "tool_name": function_name,
                            "status": "started"
                        }

                # Function call completed
                elif event.data.type == "response.output_item.done":
                    if hasattr(event.data.item, "call_id"):
                        call_id = getattr(event.data.item, "call_id", None)
                        if call_id in function_calls_active:
                            function_name = function_calls_active[call_id]

                            # Get function output
                            output_str = getattr(event.data.item, "output", "{}")
                            try:
                                output_data = json.loads(output_str) if isinstance(output_str, str) else output_str
                            except:
                                output_data = {"output": str(output_str)}

                            yield {
                                "type": "tool_execution",
                                "tool_name": function_name,
                                "status": "completed",
                                "output": output_data
                            }

                            # If artifact was created, emit artifact_created event
                            if output_data.get("success") and output_data.get("artifact_id"):
                                artifact = get_artifact_by_id(db, output_data["artifact_id"])
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

                            del function_calls_active[call_id]

        # Signal completion
        yield {"type": "response_complete"}

        logger.info("✅ Agent run completed successfully")

    except Exception as e:
        logger.error(f"❌ Agent run failed: {type(e).__name__}: {e}", exc_info=True)
        yield {
            "type": "error",
            "error": f"Agent error: {str(e)}"
        }
