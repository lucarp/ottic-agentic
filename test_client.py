#!/usr/bin/env python3
"""WebSocket test client for testing the agentic system."""

import asyncio
import json
import websockets


async def test_cop_request():
    """Test the COP CSV request."""
    uri = "ws://localhost:8000/ws"

    print("🔌 Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")

        # Receive welcome message
        welcome = await websocket.recv()
        print(f"📨 {json.loads(welcome)['content'][:50]}...")

        # Send COP request
        request = {
            "type": "user_message",
            "content": "I want a list of all Conferences of Parties (COP), from 1 to 30. Including venue, year, and main topics. Create a CSV artifact."
        }

        print(f"\n📤 Sending request: {request['content'][:80]}...")
        await websocket.send(json.dumps(request))

        # Receive and display all responses
        print("\n" + "="*80)
        print("AGENT RESPONSES:")
        print("="*80 + "\n")

        message_count = 0
        artifact_created = False

        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=180.0)
                message = json.loads(response)
                message_count += 1

                msg_type = message.get("type")

                if msg_type == "user_message":
                    print(f"👤 USER: {message.get('content', '')[:80]}")

                elif msg_type == "text_delta":
                    # Print text deltas inline
                    delta = message.get("delta", "")
                    print(delta, end="", flush=True)

                elif msg_type == "assistant_message":
                    print(f"🤖 ASSISTANT: {message.get('content', '')}")

                elif msg_type == "tool_execution":
                    tool = message.get("tool_name")
                    status = message.get("status")
                    print(f"\n🔧 TOOL: {tool} - {status}")
                    if status == "completed":
                        output = message.get("output", {})
                        if output.get("success"):
                            print(f"   ✅ {output.get('message')}")

                elif msg_type == "artifact_created":
                    artifact_created = True
                    artifact = message.get("artifact", {})
                    print(f"\n\n🎨 ARTIFACT CREATED!")
                    print(f"   ID: {artifact.get('id')}")
                    print(f"   Type: {artifact.get('type')}")
                    print(f"   Status: {artifact.get('status')}")

                    # Show CSV data preview
                    data = artifact.get("data", {})
                    if data.get("headers"):
                        print(f"\n   📊 CSV Preview:")
                        print(f"   Headers: {', '.join(data['headers'])}")
                        print(f"   Rows: {len(data.get('rows', []))} rows")
                        if data.get("rows"):
                            print(f"   First row: {data['rows'][0]}")

                elif msg_type == "error":
                    error = message.get("error")
                    print(f"\n❌ ERROR: {error}")
                    break

                elif msg_type == "response_complete":
                    print(f"\n\n✅ RESPONSE COMPLETE (received {message_count} messages)")
                    break

        except asyncio.TimeoutError:
            print(f"\n\n⏱️  TIMEOUT after 180 seconds (received {message_count} messages)")

        print("\n" + "="*80)

        if artifact_created:
            print("✅ TEST PASSED - Artifact created successfully!")
        else:
            print("⚠️  TEST INCOMPLETE - No artifact was created")

        return artifact_created


if __name__ == "__main__":
    print("Starting COP CSV Test")
    print("="*80)
    result = asyncio.run(test_cop_request())
    exit(0 if result else 1)
