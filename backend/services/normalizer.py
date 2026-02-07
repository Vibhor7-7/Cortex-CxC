"""
Conversation normalization service.

This module handles:
- Combining parsed messages into conversation objects
- Generating conversation titles from first user message
- Calculating message statistics
- Cleaning and validating message content
"""

import re
from typing import Dict, List, Any
from datetime import datetime


def normalize_conversation(
    parsed_data: Dict[str, Any],
    messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Normalize a parsed conversation into a standard format.
    
    Args:
        parsed_data: Dictionary containing conversation metadata
                    (title, timestamp, etc.)
        messages: List of message dictionaries with role, content,
                 sequence_number
    
    Returns:
        Dictionary with normalized conversation data:
        - title: str
        - messages: List[Dict]
        - message_count: int
        - created_at: datetime
        - metadata: Dict (any additional info)
    
    Raises:
        ValueError: If messages list is empty or invalid
    """
    if not messages:
        raise ValueError("Cannot normalize conversation: messages list is empty")
    
    # Clean and validate messages
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = _clean_message(msg)
        if cleaned_msg:
            cleaned_messages.append(cleaned_msg)
    
    if not cleaned_messages:
        raise ValueError("No valid messages after cleaning")
    
    # Generate or extract title
    title = _generate_title(parsed_data.get("title"), cleaned_messages)
    
    # Calculate statistics
    message_count = len(cleaned_messages)
    user_messages = sum(1 for m in cleaned_messages if m["role"] == "user")
    assistant_messages = sum(1 for m in cleaned_messages if m["role"] == "assistant")
    
    # Extract or generate timestamp
    created_at = _extract_timestamp(parsed_data.get("timestamp"))
    
    # Build normalized conversation
    normalized = {
        "title": title,
        "messages": cleaned_messages,
        "message_count": message_count,
        "created_at": created_at,
        "metadata": {
            "user_message_count": user_messages,
            "assistant_message_count": assistant_messages,
            "original_title": parsed_data.get("title"),
            "original_timestamp": parsed_data.get("timestamp"),
        }
    }
    
    return normalized


def _clean_message(message: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Clean and validate a single message.
    
    Args:
        message: Message dictionary with role, content, sequence_number
    
    Returns:
        Cleaned message dictionary or None if invalid
    """
    if not isinstance(message, dict):
        return None
    
    role = message.get("role", "").lower()
    content = message.get("content", "")
    sequence_number = message.get("sequence_number", 0)
    
    # Validate role
    if role not in ["user", "assistant", "system"]:
        return None
    
    # Clean content
    if not content or not isinstance(content, str):
        return None
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Skip empty messages
    if not content:
        return None
    
    return {
        "role": role,
        "content": content,
        "sequence_number": sequence_number
    }


def _generate_title(existing_title: str | None, messages: List[Dict[str, Any]]) -> str:
    """
    Generate a conversation title from the first user message if needed.
    
    Args:
        existing_title: Existing title from parsed data (may be None)
        messages: List of cleaned messages
    
    Returns:
        Generated or existing title (max 200 chars)
    """
    # Use existing title if valid
    if existing_title and isinstance(existing_title, str):
        title = existing_title.strip()
        if title and title != "Untitled":
            return title[:200]
    
    # Find first user message
    first_user_message = next(
        (m for m in messages if m["role"] == "user"),
        None
    )
    
    if not first_user_message:
        return "Untitled Conversation"
    
    # Extract first 50 chars of first user message
    content = first_user_message["content"]
    
    # Remove newlines and extra spaces
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Truncate to 50 chars and add ellipsis if needed
    if len(content) > 50:
        title = content[:47] + "..."
    else:
        title = content
    
    return title


def _extract_timestamp(timestamp_str: str | None) -> datetime:
    """
    Extract or generate a timestamp for the conversation.
    
    Args:
        timestamp_str: Timestamp string from parsed data
    
    Returns:
        datetime object (current time if parsing fails)
    """
    if not timestamp_str:
        return datetime.utcnow()
    
    # Try common timestamp formats
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",  # 2026-02-03 02:01:51.954688
        "%Y-%m-%d %H:%M:%S",      # 2026-02-03 02:01:51
        "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%SZ",     # ISO format
        "%Y-%m-%d",               # Date only
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # If all parsing fails, return current time
    return datetime.utcnow()


def validate_normalized_conversation(conversation: Dict[str, Any]) -> bool:
    """
    Validate that a normalized conversation has all required fields.
    
    Args:
        conversation: Normalized conversation dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["title", "messages", "message_count", "created_at"]
    
    # Check required fields exist
    for field in required_fields:
        if field not in conversation:
            return False
    
    # Validate types
    if not isinstance(conversation["title"], str):
        return False
    
    if not isinstance(conversation["messages"], list):
        return False
    
    if not isinstance(conversation["message_count"], int):
        return False
    
    if not isinstance(conversation["created_at"], datetime):
        return False
    
    # Validate message count matches
    if conversation["message_count"] != len(conversation["messages"]):
        return False
    
    return True
