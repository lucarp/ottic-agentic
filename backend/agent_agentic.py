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
from agents import Agent, Runner, function_tool, WebSearchTool, ModelSettings
from agents.exceptions import MaxTurnsExceeded
from openai.types.shared.reasoning import Reasoning
from sqlalchemy.orm import Session

from database import create_artifact, ArtifactStatus, get_artifact_by_id
from schemas import (
    CsvArtifact,
    HtmlArtifact,
    ChartArtifact,
    ChartDataset,
    PaymentLinkArtifact,
    MarkdownArtifact,
    CodeArtifact,
    DomainOverviewArtifact,
    CompetitorAnalysisArtifact,
    CompetitorData,
    KeywordResearchArtifact,
    KeywordData,
)
from seranking_client import get_seranking_client

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
    """Create artifact tools with injected database session.

    CRITICAL FIX NOTES (to prevent future errors):
    ==================================================
    The openai-agents SDK (v0.5.0+) enforces STRICT JSON SCHEMA mode for function_tool.
    This means:

    1. NEVER use @function_tool(strict=False) - this parameter doesn't exist!
    2. NEVER use Dict[str, Any] or list[Any] types - causes "additionalProperties" errors
    3. ALWAYS use concrete types: list[str], list[float], list[list[str]], etc.
    4. NEVER use Config class with extra="allow" in Pydantic models
    5. AVOID complex union types like Optional[str | list[str]] - use single types
    6. ALL Pydantic fields must generate strict JSON schemas with additionalProperties=false

    If you get "additionalProperties should not be set for object types" error:
    - Check for Any, Dict, or generic types in function signatures
    - Check for union types (|) that aren't just Optional
    - Check for Pydantic Config with extra="allow"
    - Replace with concrete, specific types

    References:
    - OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
    - Forum discussion: https://community.openai.com/t/agent-sdk-throws-error-additionalproperties-should-not-be-set-for-object-types
    """

    @function_tool
    def create_csv_artifact(
        headers: Annotated[list[str], "Column headers for the CSV"],
        rows: Annotated[list[list[str]], "Data rows as list of lists (all values as strings)"],
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
        datasets: Annotated[list[ChartDataset], "Chart.js datasets with label, data, and styling"],
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
    ) -> str:
        """Create a Stripe payment link artifact."""
        try:
            artifact_data = PaymentLinkArtifact(
                amount=amount,
                currency=currency,
                description=description,
                success_message=success_message,
                metadata=None
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

    # ========================================================================
    # SEO Analysis Tools - SE Ranking API Integration
    # ========================================================================

    @function_tool
    async def analyze_domain_seo(
        domain: Annotated[str, "Domain to analyze (e.g., 'stripe.com', 'shopify.com')"],
        currency: Annotated[str, "Currency code for traffic values (e.g., 'USD', 'EUR')"] = "USD",
    ) -> str:
        """Analyze a domain's SEO performance including organic/paid traffic, keywords, and estimated traffic value."""
        try:
            # Get SE Ranking client and fetch domain overview
            seranking = get_seranking_client()
            api_data = await seranking.get_domain_overview(domain, currency)

            # Extract data from API response
            # SE Ranking API returns arrays for organic and adv (paid) data
            organic_list = api_data.get("organic", [])
            paid_list = api_data.get("adv", [])  # Note: API uses "adv" not "paid"

            # Get first item from each list (worldwide data)
            organic_data = organic_list[0] if organic_list else {}
            paid_data = paid_list[0] if paid_list else {}

            artifact_data = DomainOverviewArtifact(
                domain=domain,
                total_keywords=organic_data.get("keywords_count", 0),
                organic_traffic=organic_data.get("traffic_sum", 0),
                organic_traffic_value=float(organic_data.get("price_sum", 0)),
                paid_keywords=paid_data.get("keywords_count", 0),
                paid_traffic=paid_data.get("traffic_sum", 0),
                paid_traffic_value=float(paid_data.get("price_sum", 0)),
                currency=currency,
                title=f"SEO Overview: {domain}"
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="domain_overview",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "domain_overview",
                "message": f"Successfully analyzed SEO performance for {domain}"
            })
        except Exception as e:
            logger.error(f"Error analyzing domain SEO: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    async def analyze_competitors(
        domain: Annotated[str, "Target domain to find competitors for (e.g., 'stripe.com')"],
        source: Annotated[str, "Regional database - country code (e.g., 'us', 'uk', 'ca', 'de')"] = "us",
        type: Annotated[str, "Analysis type: 'organic' or 'paid'"] = "organic",
        limit: Annotated[int, "Maximum number of competitors to return (1-100)"] = 10,
    ) -> str:
        """Find and analyze top SEO competitors for a domain, showing which domains compete for the same keywords."""
        try:
            # Get SE Ranking client and fetch competitors
            seranking = get_seranking_client()
            api_data = await seranking.get_competitors(domain, source, type, limit)

            # SE Ranking returns a list directly, not a dict with "competitors" key
            competitors_list = api_data if isinstance(api_data, list) else []

            # Transform to our schema format
            competitors = [
                CompetitorData(
                    rank=idx + 1,
                    domain=comp.get("domain", ""),
                    common_keywords=comp.get("common_keywords", 0)
                )
                for idx, comp in enumerate(competitors_list)
            ]

            artifact_data = CompetitorAnalysisArtifact(
                target_domain=domain,
                source=source,
                type=type,
                competitors=competitors,
                total_competitors=len(competitors),
                title=f"Competitor Analysis: {domain} ({source.upper()})"
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="competitor_analysis",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "competitor_analysis",
                "message": f"Found {len(competitors)} competitors for {domain}"
            })
        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @function_tool
    async def research_keywords(
        analysis_type: Annotated[str, "Type of analysis: 'similar' (related keywords), 'gap' (competitor keywords you don't rank for)"],
        keyword: Annotated[str | None, "Seed keyword for 'similar' analysis (e.g., 'payment gateway')"] = None,
        target_domain: Annotated[str | None, "Your domain for 'gap' analysis (e.g., 'stripe.com')"] = None,
        competitor_domain: Annotated[str | None, "Competitor domain for 'gap' analysis (e.g., 'paypal.com')"] = None,
        source: Annotated[str, "Regional database - country code (e.g., 'us', 'uk', 'ca')"] = "us",
        limit: Annotated[int, "Maximum number of keywords to return (1-100)"] = 50,
    ) -> str:
        """Research keyword opportunities: find similar keywords to expand content, or identify competitor keyword gaps."""
        try:
            seranking = get_seranking_client()

            if analysis_type == "similar":
                if not keyword:
                    return json.dumps({"success": False, "error": "keyword parameter required for similar analysis"})

                # Get similar keywords
                api_data = await seranking.get_similar_keywords(keyword, source, limit)
                keywords_list = api_data.get("keywords", [])

                # Transform to our schema
                keywords = [
                    KeywordData(
                        keyword=kw.get("keyword", ""),
                        volume=kw.get("volume", 0),
                        cpc=float(kw.get("cpc", 0)),
                        difficulty=kw.get("difficulty", 0),
                        position=None
                    )
                    for kw in keywords_list
                ]

                title = f"Similar Keywords: {keyword}"

            elif analysis_type == "gap":
                if not target_domain or not competitor_domain:
                    return json.dumps({
                        "success": False,
                        "error": "target_domain and competitor_domain required for gap analysis"
                    })

                # Get keyword gaps (keywords competitor ranks for that you don't)
                api_data = await seranking.get_keyword_comparison(
                    domain=competitor_domain,  # Keywords THEY rank for
                    competitor=target_domain,  # That YOU don't rank for
                    source=source,
                    diff=1,  # Only differences
                    limit=limit
                )
                # SE Ranking returns a list directly for keyword_comparison
                keywords_list = api_data if isinstance(api_data, list) else []

                # Transform to our schema
                keywords = [
                    KeywordData(
                        keyword=kw.get("keyword", ""),
                        volume=kw.get("volume", 0),
                        cpc=float(kw.get("cpc", 0)),
                        difficulty=kw.get("difficulty", 0),
                        position=kw.get("position")  # Competitor's position
                    )
                    for kw in keywords_list
                ]

                title = f"Keyword Gap: {competitor_domain} vs {target_domain}"

            else:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid analysis_type: {analysis_type}. Use 'similar' or 'gap'"
                })

            artifact_data = KeywordResearchArtifact(
                analysis_type=analysis_type,
                primary_keyword=keyword,
                target_domain=target_domain,
                competitor_domain=competitor_domain,
                source=source,
                keywords=keywords,
                total_results=len(keywords),
                title=title
            ).model_dump()

            artifact = create_artifact(
                db=db_session,
                artifact_type="keyword_research",
                data=artifact_data,
                status=ArtifactStatus.FINAL
            )

            return json.dumps({
                "success": True,
                "artifact_id": str(artifact.id),
                "type": "keyword_research",
                "message": f"Found {len(keywords)} keywords for {analysis_type} analysis"
            })
        except Exception as e:
            logger.error(f"Error researching keywords: {e}")
            return json.dumps({"success": False, "error": str(e)})

    # Return all tools
    return [
        create_csv_artifact,
        create_chart_artifact,
        create_markdown_artifact,
        create_code_artifact,
        create_html_artifact,
        create_payment_link_artifact,
        analyze_domain_seo,
        analyze_competitors,
        research_keywords,
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
- SEO analysis tools powered by SE Ranking API:
  * Domain SEO overview (traffic, keywords, value estimates)
  * Competitor analysis (find competing domains and keyword overlaps)
  * Keyword research (similar keywords, competitor keyword gaps)

When users request information or data:
1. Use web search if you need current information
2. Fetch URLs if you need to read specific webpages
3. For SEO-related requests, use the SEO analysis tools to provide data-driven insights
4. Create appropriate artifacts to present the information clearly
5. Always create artifacts in the user's requested language
6. Provide clear, well-structured outputs

Be helpful, accurate, and create high-quality artifacts.""",
        tools=tools,
        model="gpt-5",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium", summary="detailed"),
            verbosity="low",
        ),
    )

    return agent


# ============================================================================
# Streaming Runner
# ============================================================================

async def run_agent_agentic(
    user_input: str,
    db: Session,
    conversation_history: list[dict[str, str]] | None = None,
    previous_response_id: str | None = None
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agent with streaming support for the frontend.

    Yields events compatible with the existing WebSocket frontend.

    Args:
        user_input: The user's input message
        db: Database session
        conversation_history: Optional conversation history
        previous_response_id: Optional response ID to continue from (for max_turns exceeded)
    """
    logger.info(f"🤖 Starting Agent run for: '{user_input[:50]}...'")
    if previous_response_id:
        logger.info(f"📄 Continuing from previous response: {previous_response_id}")

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
        result = Runner.run_streamed(
            agent,
            full_input,
            previous_response_id=previous_response_id
        )

        current_text_delta = ""
        function_calls_active = {}
        last_response_id = None

        async for event in result.stream_events():
            event_type = event.type

            # Debug logging (can be disabled in production)
            if event_type == "raw_response_event" and hasattr(event, 'data') and hasattr(event.data, 'type'):
                logger.info(f"🔍 Raw event data type: {event.data.type}")

            # Handle raw response events for reasoning and text deltas
            if event_type == "raw_response_event":
                # Check if this is a reasoning or text delta
                if hasattr(event, 'data') and hasattr(event.data, 'type'):
                    if event.data.type == "response.reasoning_summary_text.delta":
                        # Stream reasoning tokens to frontend
                        yield {
                            "type": "reasoning_delta",
                            "delta": event.data.delta
                        }
                        continue
                    # Skip other raw response events
                continue

            # Handle run item stream events (this is the correct way per openai-agents docs)
            elif event_type == "run_item_stream_event":
                item = event.item

                if item.type == "tool_call_item":
                    # Tool call started - access raw_item for function name and call_id
                    raw_item = getattr(item, "raw_item", None)

                    # CRITICAL FIX: raw_item is ResponseFunctionToolCall object, use attribute access
                    if raw_item:
                        function_name = getattr(raw_item, "name", "unknown")
                        call_id = getattr(raw_item, "call_id", "unknown")
                        function_calls_active[call_id] = function_name

                        logger.info(f"🔧 Tool call started: {function_name}, call_id={call_id}")

                        yield {
                            "type": "tool_execution",
                            "tool_name": function_name,
                            "status": "started"
                        }

                elif item.type == "tool_call_output_item":
                    # Tool call output - use event.item.output directly per docs
                    output_str = getattr(item, "output", "{}")

                    # Get call_id from raw_item - handle both dict and object formats
                    raw_item = getattr(item, "raw_item", None)
                    if isinstance(raw_item, dict):
                        call_id = raw_item.get("call_id")
                    else:
                        call_id = getattr(raw_item, "call_id", None) if raw_item else None

                    logger.info(f"✅ Tool call output received: call_id={call_id}, function={function_calls_active.get(call_id, 'unknown')}")

                    if call_id and call_id in function_calls_active:
                        function_name = function_calls_active[call_id]

                        try:
                            output_data = json.loads(output_str) if isinstance(output_str, str) else output_str
                        except Exception as parse_err:
                            logger.error(f"❌ Failed to parse output: {parse_err}")
                            output_data = {"output": str(output_str)}

                        logger.info(f"✅ Tool output for {function_name}: {output_data}")

                        yield {
                            "type": "tool_execution",
                            "tool_name": function_name,
                            "status": "completed",
                            "output": output_data
                        }

                        # If artifact was created, emit artifact_created event
                        if output_data.get("success") and output_data.get("artifact_id"):
                            logger.info(f"📦 Emitting artifact_created event for artifact_id: {output_data.get('artifact_id')}")
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

                elif item.type == "message_output_item":
                    # Assistant message output - use ItemHelpers per docs
                    from agents import ItemHelpers
                    text_content = ItemHelpers.text_message_output(item)
                    if text_content:
                        yield {
                            "type": "assistant_message",
                            "content": text_content
                        }

        # Signal completion
        yield {"type": "response_complete"}

        logger.info("✅ Agent run completed successfully")

    except MaxTurnsExceeded as e:
        # Agent hit max turns limit - offer to continue
        logger.warning(f"⚠️ Max turns exceeded - offering continue option")
        last_response_id = result.last_response_id if hasattr(result, 'last_response_id') else None
        yield {
            "type": "max_turns_exceeded",
            "message": "The agent has reached the maximum number of thinking rounds (10). Would you like to continue?",
            "response_id": last_response_id
        }

    except Exception as e:
        logger.error(f"❌ Agent run failed: {type(e).__name__}: {e}", exc_info=True)
        yield {
            "type": "error",
            "error": f"Agent error: {str(e)}"
        }
