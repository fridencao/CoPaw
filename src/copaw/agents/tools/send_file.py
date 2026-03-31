# -*- coding: utf-8 -*-
# flake8: noqa: E501
# pylint: disable=line-too-long,too-many-return-statements
import os
import mimetypes
import unicodedata

from .tool_types import ToolResponse, text_content

from ..schema import FileBlock


def _auto_as_type(mt: str) -> str:
    if mt.startswith("image/"):
        return "image"
    if mt.startswith("audio/"):
        return "audio"
    if mt.startswith("video/"):
        return "video"
    return "file"


async def send_file_to_user(
    file_path: str,
) -> ToolResponse:
    """Send a file to the user.

    Args:
        file_path (`str`):
            Path to the file to send.

    Returns:
        `ToolResponse`:
            The tool response containing the file or an error message.
    """

    # Normalize the path: expand ~ and fix Unicode normalization differences
    # (e.g. macOS stores filenames as NFD but paths from the LLM arrive as NFC,
    # causing os.path.exists to return False for files that do exist).
    file_path = os.path.expanduser(unicodedata.normalize("NFC", file_path))

    if not os.path.exists(file_path):
        return ToolResponse(
            content=text_content(f"Error: The file {file_path} does not exist."),
        )

    if not os.path.isfile(file_path):
        return ToolResponse(
            content=text_content(f"Error: The path {file_path} is not a file."),
        )

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Default to application/octet-stream for unknown types
        mime_type = "application/octet-stream"
    as_type = _auto_as_type(mime_type)

    try:
        # Use local file URL instead of base64
        absolute_path = os.path.abspath(file_path)
        file_url = f"file://{absolute_path}"
        source = {"type": "url", "url": file_url}

        content_parts = []

        if as_type == "image":
            content_parts.append({"type": "image_url", "image_url": {"url": file_url}})
        elif as_type == "audio":
            content_parts.append({"type": "audio", "audio": file_url})
        elif as_type == "video":
            content_parts.append({"type": "video", "video": file_url})
        else:
            content_parts.append({
                "type": "file",
                "source": source,
                "filename": os.path.basename(file_path),
            })

        content_parts.append({"type": "text", "text": "File sent successfully."})

        return ToolResponse(content=content_parts)

    except Exception as e:
        return ToolResponse(
            content=text_content(f"Error: Send file failed due to \n{e}"),
        )