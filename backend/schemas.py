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
    """CSV data artifact with headers and rows."""
    headers: list[str] = Field(description="Column headers for the CSV")
    rows: list[list[Any]] = Field(description="Data rows as list of lists")
    title: Optional[str] = Field(default=None, description="Optional title for the CSV")
    description: Optional[str] = Field(default=None, description="Optional description")


class HtmlArtifact(BaseModel):
    """HTML content artifact."""
    html: str = Field(description="The HTML content to render")
    title: Optional[str] = Field(default=None, description="Optional title")
    css: Optional[str] = Field(default=None, description="Optional custom CSS styles")


class ChartArtifact(BaseModel):
    """Chart/visualization artifact using Chart.js format."""
    chart_type: str = Field(description="Type of chart: 'bar', 'line', 'pie', 'doughnut', 'scatter', etc.")
    labels: list[str] = Field(description="Labels for the chart data points")
    datasets: list[dict[str, Any]] = Field(description="Chart.js datasets with label, data, and styling")
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
