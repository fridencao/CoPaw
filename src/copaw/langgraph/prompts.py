# -*- coding: utf-8 -*-
"""System prompt builder for LangGraph agent.

This module provides utilities for building the system prompt
from CoPaw's agent configuration files (AGENTS.md, SOUL.md, PROFILE.md).
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default system prompt when no config files are found
DEFAULT_SYSTEM_PROMPT = """You are CoPaw, a helpful AI assistant.

You have access to various tools to help you accomplish tasks.
When you need to use a tool, make sure to use it properly.
After using a tool, you will receive the results and can continue.
"""


def build_system_prompt(
    agent_config: Any,
    workspace_dir: Optional[Path] = None,
) -> str:
    """Build system prompt from agent configuration files.

    This function reads from the following files in order:
    - AGENTS.md: Agent definition and personality
    - SOUL.md: Agent's core values and principles
    - PROFILE.md: Additional profile information
    - instructions.md: Additional instructions

    Args:
        agent_config: Agent configuration object
        workspace_dir: Workspace directory containing config files

    Returns:
        Complete system prompt string
    """
    if workspace_dir is None:
        from ...constant import WORKING_DIR
        workspace_dir = WORKING_DIR / "agents" / agent_config.id

    prompt_parts = []

    # 1. Read AGENTS.md - Agent definition and personality
    agents_md = workspace_dir / "AGENTS.md"
    if agents_md.exists():
        try:
            content = agents_md.read_text(encoding="utf-8").strip()
            if content:
                prompt_parts.append(f"# Agent Definition\n{content}")
        except Exception as e:
            logger.warning(f"Failed to read AGENTS.md: {e}")

    # 2. Read SOUL.md - Core values and principles
    soul_md = workspace_dir / "SOUL.md"
    if soul_md.exists():
        try:
            content = soul_md.read_text(encoding="utf-8").strip()
            if content:
                prompt_parts.append(f"# Agent Soul\n{content}")
        except Exception as e:
            logger.warning(f"Failed to read SOUL.md: {e}")

    # 3. Read PROFILE.md - Additional profile information
    profile_md = workspace_dir / "PROFILE.md"
    if profile_md.exists():
        try:
            content = profile_md.read_text(encoding="utf-8").strip()
            if content:
                prompt_parts.append(f"# Profile\n{content}")
        except Exception as e:
            logger.warning(f"Failed to read PROFILE.md: {e}")

    # 4. Read instructions.md - Additional instructions
    instructions_md = workspace_dir / "instructions.md"
    if instructions_md.exists():
        try:
            content = instructions_md.read_text(encoding="utf-8").strip()
            if content:
                prompt_parts.append(f"# Instructions\n{content}")
        except Exception as e:
            logger.warning(f"Failed to read instructions.md: {e}")

    # 5. Add running configuration if available
    if hasattr(agent_config, "running"):
        running = agent_config.running
        if hasattr(running, "system_prompt") and running.system_prompt:
            prompt_parts.append(f"# Custom System Prompt\n{running.system_prompt}")

    # Combine all parts or use default
    if prompt_parts:
        system_prompt = "\n\n".join(prompt_parts)
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    logger.debug(
        "Built system prompt for agent '%s' (length: %d)",
        agent_config.id,
        len(system_prompt),
    )

    return system_prompt


def build_system_prompt_from_files(
    workspace_dir: Path,
    custom_prompt: Optional[str] = None,
) -> str:
    """Build system prompt from workspace files.

    This is a simplified version that takes only the workspace directory.

    Args:
        workspace_dir: Workspace directory containing config files
        custom_prompt: Optional custom prompt to append

    Returns:
        Complete system prompt string
    """
    # Create a minimal config object
    class MinimalConfig:
        id = "default"
        running = type('obj', (object,), {'system_prompt': custom_prompt})()

    return build_system_prompt(MinimalConfig(), workspace_dir)


__all__ = [
    "build_system_prompt",
    "build_system_prompt_from_files",
    "DEFAULT_SYSTEM_PROMPT",
]