"""GPT-5 agent with typed artifact tools."""

import json
import os
import sys
from typing import Any, AsyncGenerator, Optional

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
)


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool definitions for GPT-5 Responses API
TOOLS = [
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
]


def execute_artifact_tool(tool_name: str, tool_input: dict[str, Any], db: Session) -> dict[str, Any]:
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

        # Create initial response with GPT-5
        try:
            response = await client.responses.create(
                model="gpt-5",
                input=user_input,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
                tools=TOOLS,
                previous_response_id=previous_response_id,
                stream=False
            )
            sys.stderr.write(f"GPT-5 response received successfully\n")
            sys.stderr.flush()
        except Exception as api_error:
            sys.stderr.write(f"\n=== GPT-5 API ERROR ===\n")
            sys.stderr.write(f"Error type: {type(api_error).__name__}\n")
            sys.stderr.write(f"Error message: {str(api_error)}\n")
            sys.stderr.flush()
            raise

        response_id = response.id
        sys.stderr.write(f"Response ID: {response_id}\n")
        sys.stderr.flush()

        # Loop to handle multiple rounds of tool calling
        while True:
            tool_outputs = []
            has_function_calls = False

            sys.stderr.write(f"Processing response with {len(response.output)} output items\n")
            sys.stderr.flush()

            # Process the response
            for idx, item in enumerate(response.output):
                sys.stderr.write(f"Processing item {idx}: type={item.type}\n")
                sys.stderr.flush()
                # Handle reasoning output
                if item.type == "reasoning":
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
                    result = execute_artifact_tool(tool_name, tool_input, db)

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
