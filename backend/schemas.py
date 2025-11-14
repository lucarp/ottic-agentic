"""Pydantic schemas for API responses and tool definitions."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class ArtifactStatus(str, Enum):
    """Status of an artifact."""
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class ArtifactResponse(BaseModel):
    """Response schema for artifact."""
    id: UUID
    type: str
    status: ArtifactStatus
    data: dict[str, Any]
    artifact_metadata: Optional[dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Artifact Type Schemas - Each is a separate tool with structured output
# ============================================================================

class CsvArtifact(BaseModel):
    """CSV data artifact with headers and rows.

    IMPORTANT: For openai-agents strict JSON schema compatibility:
    - Use concrete types only (str, int, float, bool, list[str], etc.)
    - NEVER use Any, Dict[str, Any], or list[Any] - these cause additionalProperties errors
    - All nested lists must have concrete types: list[list[str]] not list[list[Any]]
    - For mixed-type data, convert to strings at the tool level
    """
    headers: list[str] = Field(description="Column headers for the CSV")
    rows: list[list[str]] = Field(description="Data rows as list of lists (all values as strings)")
    title: Optional[str] = Field(default=None, description="Optional title for the CSV")
    description: Optional[str] = Field(default=None, description="Optional description")


class HtmlArtifact(BaseModel):
    """HTML content artifact."""
    html: str = Field(description="The HTML content to render")
    title: Optional[str] = Field(default=None, description="Optional title")
    css: Optional[str] = Field(default=None, description="Optional custom CSS styles")


class ChartDataset(BaseModel):
    """Chart.js dataset structure.

    IMPORTANT: For strict JSON schema compatibility:
    - Avoid union types like Optional[str | list[str]] - use single types only
    - Do NOT use Config class with extra="allow" - causes additionalProperties errors
    - Define all expected fields explicitly rather than allowing arbitrary properties
    """
    label: str = Field(description="Dataset label")
    data: list[float] = Field(description="Data values")
    backgroundColor: Optional[str] = Field(default=None, description="Background color")
    borderColor: Optional[str] = Field(default=None, description="Border color")
    borderWidth: Optional[int] = Field(default=None, description="Border width")
    fill: Optional[bool] = Field(default=None, description="Fill under line (for line charts)")
    tension: Optional[float] = Field(default=None, description="Line tension/curve (for line charts)")


class ChartArtifact(BaseModel):
    """Chart/visualization artifact using Chart.js format."""
    chart_type: str = Field(description="Type of chart: 'bar', 'line', 'pie', 'doughnut', 'scatter', etc.")
    labels: list[str] = Field(description="Labels for the chart data points")
    datasets: list[ChartDataset] = Field(description="Chart.js datasets with label, data, and styling")
    title: Optional[str] = Field(default=None, description="Chart title")
    x_axis_label: Optional[str] = Field(default=None, description="X-axis label")
    y_axis_label: Optional[str] = Field(default=None, description="Y-axis label")


class PaymentLinkArtifact(BaseModel):
    """Stripe payment link artifact."""
    amount: float = Field(description="Payment amount in USD", gt=0)
    currency: str = Field(default="usd", description="Currency code (e.g., 'usd', 'eur')")
    description: str = Field(description="Payment description")
    success_message: Optional[str] = Field(default=None, description="Message shown on success")
    metadata: Optional[dict[str, str]] = Field(default=None, description="Additional metadata")


class MarkdownArtifact(BaseModel):
    """Markdown document artifact."""
    content: str = Field(description="Markdown formatted content")
    title: Optional[str] = Field(default=None, description="Document title")


class CodeArtifact(BaseModel):
    """Code snippet artifact with syntax highlighting."""
    code: str = Field(description="The code content")
    language: str = Field(description="Programming language (e.g., 'python', 'javascript', 'sql')")
    title: Optional[str] = Field(default=None, description="Code snippet title")
    description: Optional[str] = Field(default=None, description="Description of what the code does")


class FetchedLink(BaseModel):
    """Content fetched from a URL."""
    url: str = Field(description="The URL that was fetched")
    title: Optional[str] = Field(default=None, description="Page title extracted from content")
    content: str = Field(description="Main content extracted from the page (markdown or text)")
    content_type: str = Field(default="markdown", description="Format of the content: 'markdown', 'html', or 'text'")
    fetch_timestamp: Optional[str] = Field(default=None, description="When the content was fetched")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata (author, publish date, etc.)")
    summary: Optional[str] = Field(default=None, description="Optional AI-generated summary of the content")


# ============================================================================
# SEO Analysis Artifact Schemas - SE Ranking API Integration
# ============================================================================

class DomainOverviewArtifact(BaseModel):
    """Domain SEO overview artifact with traffic and keyword metrics.

    IMPORTANT: Uses strict typing for openai-agents compatibility.
    All numeric fields are concrete types (int, float).
    """
    domain: str = Field(description="Domain being analyzed (e.g., 'stripe.com')")
    total_keywords: int = Field(description="Total organic keywords ranking")
    organic_traffic: int = Field(description="Estimated monthly organic traffic")
    organic_traffic_value: float = Field(description="Estimated monthly value of organic traffic in specified currency")
    paid_keywords: int = Field(description="Total paid keywords")
    paid_traffic: int = Field(description="Estimated monthly paid traffic")
    paid_traffic_value: float = Field(description="Estimated monthly value of paid traffic in specified currency")
    currency: str = Field(default="USD", description="Currency code for traffic values")
    title: Optional[str] = Field(default=None, description="Optional title for the overview")


class CompetitorData(BaseModel):
    """Individual competitor data in competitor analysis.

    IMPORTANT: Nested model with strict typing - no Any types.
    """
    rank: int = Field(description="Competitor rank (1 = closest competitor)")
    domain: str = Field(description="Competitor domain name")
    common_keywords: int = Field(description="Number of keywords in common with target domain")


class CompetitorAnalysisArtifact(BaseModel):
    """Competitor analysis artifact showing top SEO competitors.

    IMPORTANT: Uses list[CompetitorData] for strict typing.
    """
    target_domain: str = Field(description="The domain being analyzed")
    source: str = Field(description="Regional database used (e.g., 'us', 'uk')")
    type: str = Field(description="Analysis type: 'organic' or 'paid'")
    competitors: list[CompetitorData] = Field(description="List of competitor domains with metrics")
    total_competitors: int = Field(description="Total number of competitors found")
    title: Optional[str] = Field(default=None, description="Optional title for the analysis")


class KeywordData(BaseModel):
    """Individual keyword data in keyword research.

    IMPORTANT: All fields are concrete types for strict JSON schema.
    """
    keyword: str = Field(description="The keyword phrase")
    volume: int = Field(description="Monthly search volume")
    cpc: float = Field(description="Cost per click in USD")
    difficulty: int = Field(description="SEO difficulty score (0-100, higher = harder)")
    position: Optional[int] = Field(default=None, description="Current ranking position (for gap analysis)")


class KeywordResearchArtifact(BaseModel):
    """Keyword research artifact with search volume and competition data.

    IMPORTANT: Uses list[KeywordData] for strict typing.
    Supports multiple analysis types: similar, related, or competitor gap analysis.
    """
    analysis_type: str = Field(description="Type of analysis: 'similar', 'related', or 'gap'")
    primary_keyword: Optional[str] = Field(default=None, description="Seed keyword for similar/related analysis")
    target_domain: Optional[str] = Field(default=None, description="Target domain for gap analysis")
    competitor_domain: Optional[str] = Field(default=None, description="Competitor domain for gap analysis")
    source: str = Field(default="us", description="Regional database (e.g., 'us', 'uk')")
    keywords: list[KeywordData] = Field(description="List of keywords with metrics")
    total_results: int = Field(description="Total number of keywords found")
    title: Optional[str] = Field(default=None, description="Optional title for the research")


class WebSearchTool(BaseModel):
    """Tool for performing web searches using Tavily AI."""
    query: str = Field(description="The search query to find relevant web information")
    max_results: int = Field(default=5, description="Maximum number of search results to return (1-10)", ge=1, le=10)


class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserMessage(WebSocketMessage):
    """User message from frontend."""
    type: str = "user_message"
    content: str


class AssistantMessage(WebSocketMessage):
    """Assistant response message."""
    type: str = "assistant_message"
    content: str


class ToolExecutionMessage(WebSocketMessage):
    """Tool execution notification."""
    type: str = "tool_execution"
    tool_name: str
    status: str  # "started", "completed", "failed"
    input: Optional[dict[str, Any]] = None
    output: Optional[Any] = None


class ArtifactCreatedMessage(WebSocketMessage):
    """Artifact creation notification."""
    type: str = "artifact_created"
    artifact: ArtifactResponse


class ErrorMessage(WebSocketMessage):
    """Error message."""
    type: str = "error"
    error: str
    details: Optional[str] = None
