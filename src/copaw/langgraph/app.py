# -*- coding: utf-8 -*-
"""Simple FastAPI app using LangGraph Runner.

This is a minimal FastAPI application that demonstrates how to use
the LangGraph-based runner directly, without the AgentScope framework.

This can be used as a reference for complete migration or as a
standalone test application.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from copaw.langgraph.adapter import LangGraphRunnerAdapter
from copaw.langgraph.tools import register_builtin_tools
from copaw.config.config import load_agent_config
from copaw.constant import WORKING_DIR

app = FastAPI(
    title="CoPaw LangGraph API",
    description="Minimal API using LangGraph-based agent",
    version="0.1.0",
)


# Store runners per agent
_runners = {}


def get_runner(agent_id: str = "default") -> LangGraphRunnerAdapter:
    """Get or create a runner for the given agent."""
    if agent_id not in _runners:
        # Get agent config
        try:
            agent_config = load_agent_config(agent_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_id}' not found: {e}",
            )

        # Create workspace directory
        workspace_dir = WORKING_DIR / "agents" / agent_id

        # Create runner adapter
        runner = LangGraphRunnerAdapter(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
        )

        _runners[agent_id] = runner

    return _runners[agent_id]


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    agent_id: str = "default"
    session_id: Optional[str] = None
    user_id: Optional[str] = "default"
    channel: str = "console"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_id: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CoPaw LangGraph API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint."""
    import uuid

    session_id = request.session_id or str(uuid.uuid4())

    # Get runner
    runner = get_runner(request.agent_id)

    # Create mock AgentRequest
    class MockRequest:
        def __init__(self):
            self.session_id = session_id
            self.user_id = request.user_id
            self.channel = request.channel

    # Execute chat
    response_text = ""
    mock_request = MockRequest()

    try:
        async for msg, last in runner.query_handler(
            msgs=[{"role": "user", "content": request.message}],
            request=mock_request,
        ):
            if hasattr(msg, "content"):
                response_text = msg.content

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            agent_id=request.agent_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    import uuid

    session_id = request.session_id or str(uuid.uuid4())
    runner = get_runner(request.agent_id)

    class MockRequest:
        def __init__(self):
            self.session_id = session_id
            self.user_id = request.user_id
            self.channel = request.channel

    async def generate():
        try:
            async for msg, last in runner.query_handler(
                msgs=[{"role": "user", "content": request.message}],
                request=MockRequest(),
            ):
                if hasattr(msg, "content"):
                    yield f"data: {msg.content}\n\n"
                if last:
                    break
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@app.get("/agents")
async def list_agents():
    """List available agents."""
    from copaw.config.utils import load_config

    try:
        config = load_config()
        agents = list(config.agents.profiles.keys())
        return {"agents": agents}
    except Exception as e:
        return {"agents": ["default"], "error": str(e)}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    from copaw.langgraph.tools import tool_registry

    tools = []
    for name in tool_registry.list_tools():
        tool = tool_registry.get_tool(name)
        tools.append({
            "name": tool.name,
            "description": tool.description,
        })

    return {"tools": tools}


if __name__ == "__main__":
    import uvicorn

    # Register built-in tools
    register_builtin_tools()

    print("Starting CoPaw LangGraph API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)