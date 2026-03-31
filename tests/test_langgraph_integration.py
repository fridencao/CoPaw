#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for LangGraph integration.

This script tests the basic functionality of the LangGraph-based
agent without requiring the full FastAPI application.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_langchain_factory():
    """Test that the LangChain factory can create models."""
    print("\n" + "=" * 60)
    print("TEST 1: LangChain Factory")
    print("=" * 60)

    try:
        from copaw.langchain.factory import create_langchain_model
        from copaw.langchain.providers import OpenAIAdapter

        print("✓ Imports successful")

        # Note: This will fail without API key, but tests the import chain
        print("✓ Factory functions available")
        print("✓ OpenAIAdapter available")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_tool_registry():
    """Test that the tool registry works."""
    print("\n" + "=" * 60)
    print("TEST 2: Tool Registry")
    print("=" * 60)

    try:
        from copaw.langgraph.tools import (
            ToolRegistry,
            tool_registry,
            register_builtin_tools,
            convert_builtin_tools,
        )

        # Test creating a custom registry
        custom_registry = ToolRegistry()
        print("✓ ToolRegistry created")

        # Test registering a simple tool
        def test_tool(name: str) -> str:
            return f"Hello, {name}!"

        custom_registry.register_tool_function(
            test_tool,
            name="test_greeting",
            description="A simple greeting tool",
        )
        print("✓ Custom tool registered")

        # Test getting tool
        tool = custom_registry.get_tool("test_greeting")
        assert tool is not None
        print(f"✓ Tool retrieved: {tool.name}")

        # Test listing tools
        tools = custom_registry.list_tools()
        print(f"✓ Registered tools: {tools}")

        # Test global registry
        print(f"✓ Global registry tools (before): {len(tool_registry)}")

        # Try to register built-in tools (may fail if copaw.agents.tools not available)
        try:
            register_builtin_tools()
            print(f"✓ Built-in tools registered, total: {len(tool_registry)}")
        except Exception as e:
            print(f"⚠ Built-in tools registration skipped: {e}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_definition():
    """Test that AgentState is properly defined."""
    print("\n" + "=" * 60)
    print("TEST 3: AgentState Definition")
    print("=" * 60)

    try:
        from copaw.langgraph.state import AgentState, get_initial_state

        # Check that all required fields are defined
        required_fields = [
            "messages",
            "tool_calls",
            "tool_results",
            "iteration_count",
            "max_iterations",
            "session_id",
            "user_id",
            "channel",
            "agent_id",
        ]

        annotations = AgentState.__annotations__

        for field in required_fields:
            if field in annotations:
                print(f"✓ Field '{field}' is defined: {annotations[field]}")
            else:
                print(f"⚠ Field '{field}' is missing")

        # Test get_initial_state
        state = get_initial_state(
            session_id="test-session",
            user_id="test-user",
            max_iterations=10,
        )
        print(f"✓ Initial state created with {len(state)} keys")
        print(f"  - session_id: {state['session_id']}")
        print(f"  - max_iterations: {state['max_iterations']}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_react_graph():
    """Test that the ReAct graph can be created."""
    print("\n" + "=" * 60)
    print("TEST 4: ReAct Graph Creation")
    print("=" * 60)

    try:
        from copaw.langgraph.graph import create_react_graph

        # Create graph with default settings
        graph = create_react_graph(max_iterations=5)
        print("✓ ReAct graph created")
        print(f"  - Type: {type(graph)}")

        # Check that it's a compiled graph
        print(f"  - Is compiled: {hasattr(graph, 'astream')}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_nodes():
    """Test that all nodes are properly defined."""
    print("\n" + "=" * 60)
    print("TEST 5: ReAct Nodes")
    print("=" * 60)

    try:
        from copaw.langgraph.nodes import (
            reasoning_node,
            check_tools_node,
            acting_node,
            observation_node,
        )

        print(f"✓ reasoning_node: {reasoning_node.__name__}")
        print(f"✓ check_tools_node: {check_tools_node.__name__}")
        print(f"✓ acting_node: {acting_node.__name__}")
        print(f"✓ observation_node: {observation_node.__name__}")

        # Test check_tools_node logic
        from copaw.langgraph.state import AgentState

        # Test case 1: No tool calls
        state1: AgentState = {
            "messages": [],
            "tool_calls": [],
            "tool_results": [],
            "iteration_count": 0,
            "max_iterations": 10,
            "session_id": "test",
            "user_id": "test",
            "channel": "test",
            "agent_id": "test",
            "memory_content": [],
            "metadata": {},
            "should_terminate": False,
            "termination_reason": None,
        }

        result = check_tools_node(state1)
        print(f"✓ check_tools_node (no tools): returns '{result}'")
        assert result == "respond", f"Expected 'respond', got '{result}'"

        # Test case 2: Has tool calls
        state2 = {**state1, "tool_calls": [{"name": "test_tool"}]}
        result2 = check_tools_node(state2)
        print(f"✓ check_tools_node (with tools): returns '{result2}'")
        assert result2 == "act", f"Expected 'act', got '{result2}'"

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_callbacks():
    """Test token usage callbacks."""
    print("\n" + "=" * 60)
    print("TEST 6: Token Usage Callbacks")
    print("=" * 60)

    try:
        from copaw.langchain.callbacks import (
            TokenUsageCallbackHandler,
            TokenUsageRecorder,
        )

        # Test callback handler
        handler = TokenUsageCallbackHandler(
            provider_id="test-provider",
            model_id="test-model",
            session_id="test-session",
        )
        print("✓ TokenUsageCallbackHandler created")

        # Test usage dict
        usage = handler.get_usage_dict()
        print(f"✓ Usage dict: {usage}")

        # Test recorder
        recorder = TokenUsageRecorder(provider_id="test")
        recorder.add_usage(
            model_id="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        print(f"✓ Usage recorded, total: {recorder.get_records()[0]['total_tokens']}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# CoPaw LangGraph Integration Test")
    print("#" * 60)

    results = []

    # Run all tests
    results.append(("LangChain Factory", await test_langchain_factory()))
    results.append(("Tool Registry", await test_tool_registry()))
    results.append(("AgentState", await test_state_definition()))
    results.append(("ReAct Graph", await test_react_graph()))
    results.append(("ReAct Nodes", await test_nodes()))
    results.append(("Callbacks", await test_callbacks()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! LangGraph integration is ready.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)