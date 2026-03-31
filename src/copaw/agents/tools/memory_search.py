# -*- coding: utf-8 -*-
"""Memory search tool for semantic search in memory files."""

from .tool_types import ToolResponse, text_content


def create_memory_search_tool(memory_manager):
    """Create a memory_search tool function with bound memory_manager.

    Args:
        memory_manager: BaseMemoryManager instance to use for searching

    Returns:
        An async function that can be registered as a tool
    """

    async def memory_search(
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> ToolResponse:
        """
        Search MEMORY.md and memory/*.md files semantically.

        Use this tool before answering questions about prior work, decisions,
        dates, people, preferences, or todos. Returns top relevant snippets
        with file paths and line numbers.

        Args:
            query (`str`):
                The semantic search query to find relevant memory snippets.
            max_results (`int`, optional):
                Maximum number of search results to return. Defaults to 5.
            min_score (`float`, optional):
                Minimum similarity score for results. Defaults to 0.1.

        Returns:
            `ToolResponse`:
                Search results formatted with paths, line numbers, and content.
        """
        if memory_manager is None:
            return ToolResponse(
                content=text_content("Error: Memory manager is not enabled."),
            )

        try:
            # memory_manager.memory_search returns search results
            result = await memory_manager.memory_search(
                query=query,
                max_results=max_results,
                min_score=min_score,
            )
            # Format results as text
            if isinstance(result, dict) and "results" in result:
                results = result["results"]
                if not results:
                    return ToolResponse(content=text_content("No relevant memories found."))

                formatted = []
                for r in results:
                    text = r.get("content", "")
                    path = r.get("path", "unknown")
                    score = r.get("score", 0)
                    formatted.append(f"[{path}] (score: {score:.2f})\n{text}\n")

                return ToolResponse(content=text_content("\n---\n".join(formatted)))
            return ToolResponse(content=text_content(str(result)))

        except Exception as e:
            return ToolResponse(
                content=text_content(f"Error: Memory search failed due to\n{e}"),
            )

    return memory_search