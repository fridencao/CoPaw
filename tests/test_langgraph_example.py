# -*- coding: utf-8 -*-
"""Example: Using LangGraph Runner directly.

This module demonstrates how to use the LangGraph-based agent
without the full FastAPI application, for testing or standalone use.
"""

import asyncio
import os
import sys
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def run_example():
    """Run a simple example using LangGraph Runner."""
    print("\n" + "=" * 60)
    print("LangGraph Runner Example")
    print("=" * 60)

    # Import required modules
    from copaw.langgraph.runner import LangGraphRunner
    from copaw.langgraph.tools import tool_registry
    from langchain_core.messages import HumanMessage

    # For testing without real API key, we'll simulate
    # In production, this would use real provider config

    print("\nNote: This example requires a configured provider.")
    print("The LangGraph infrastructure is ready.")

    # Demonstrate the structure
    print("\n--- Available Tools ---")
    for tool_name in tool_registry.list_tools():
        tool = tool_registry.get_tool(tool_name)
        print(f"  - {tool_name}: {tool.description[:50]}...")

    print("\n--- Example Configuration ---")
    print("""
# To use with a real model, configure a provider:
# 1. Set up provider in CoPaw config (providers.yaml)
# 2. Create agent with that provider
# 3. Use LangGraphRunner with agent_config

# Example code:
from copaw.langgraph.runner import LangGraphRunner

runner = LangGraphRunner(
    agent_config=my_agent_config,
    enable_builtin_tools=True,
)

# Stream responses
async for event in runner.execute(
    user_input="Hello, what can you do?",
    session_id="test-session",
):
    print(event)
""")

    return True


async def test_with_mock_model():
    """Test with a mock model to demonstrate flow."""
    print("\n" + "=" * 60)
    print("Testing with Mock Model")
    print("=" * 60)

    from copaw.langgraph.state import get_initial_state
    from copaw.langgraph.graph import create_react_graph
    from copaw.langgraph.tools import tool_registry
    from copaw.langgraph.nodes import (
        reasoning_node,
        check_tools_node,
        acting_node,
        observation_node,
    )
    from langchain_core.messages import AIMessage, HumanMessage

    # Register a simple test tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool_registry.register_tool_function(
        greet,
        name="greet",
        description="Greet a person by name",
    )

    # Create graph
    graph = create_react_graph(max_iterations=3)

    # Prepare initial state with mock model
    from unittest.mock import AsyncMock

    mock_response = AIMessage(
        content="I'll help you with that.",
        tool_calls=[
            {"id": "call_1", "name": "greet", "args": {"name": "World"}}
        ]
    )

    state = get_initial_state(
        session_id="mock-test",
        max_iterations=3,
    )
    state["messages"] = [HumanMessage(content="Say hello")]
    state["chat_model"] = AsyncMock(return_value=mock_response)
    state["tool_registry"] = tool_registry

    print(f"Initial state:")
    print(f"  - messages: {len(state['messages'])}")
    print(f"  - tool_registry has {len(tool_registry)} tools")

    # Run one iteration manually to test flow
    print("\n--- Running Reasoning Node ---")
    result_state = await reasoning_node(state)
    print(f"  - Response: {result_state['messages'][-1].content[:50]}...")
    print(f"  - Tool calls: {len(result_state['tool_calls'])}")

    if result_state['tool_calls']:
        print("\n--- Running Check Tools Node ---")
        next_step = check_tools_node(result_state)
        print(f"  - Next step: {next_step}")

        if next_step == "act":
            print("\n--- Running Acting Node ---")
            result_state = await acting_node(result_state)
            print(f"  - Tool results: {len(result_state['tool_results'])}")
            if result_state['tool_results']:
                print(f"  - Result: {result_state['tool_results'][0].get('content', '')}")

    print("\n✓ Mock test completed successfully!")
    return True


async def main():
    """Run examples."""
    await run_example()
    await test_with_mock_model()

    print("\n" + "=" * 60)
    print("Integration example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())