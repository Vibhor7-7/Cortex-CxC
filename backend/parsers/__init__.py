"""
Chat conversation HTML parsers.

This module provides parsers for various chat export formats
including ChatGPT and Claude conversations.

Usage:
    from backend.parsers import parse_html, detect_format
    
    # Auto-detect and parse
    result = parse_html(html_content)
    
    # Or detect format first
    format_type = detect_format(html_content)
    if format_type == 'chatgpt':
        from backend.parsers import ChatGPTParser
        parser = ChatGPTParser(html_content)
        result = parser.parse()
"""

from backend.parsers.base_parser import BaseParser, ParserFactory
from backend.parsers.chatgpt_parser import ChatGPTParser
from backend.parsers.claude_parser import ClaudeParser
from typing import Optional, Dict, List


def parse_html(html_content: str) -> Optional[Dict]:
    """
    Auto-detect format and parse HTML chat export.
    
    Args:
        html_content: Raw HTML string of the conversation export
        
    Returns:
        Dict with parsed conversation data:
            - title: str
            - messages: List[Dict] with role, content, sequence_number
            - created_at: Optional[datetime]
        
        Returns None if format cannot be detected.
    
    Example:
        >>> html = open('chatgpt_export.html').read()
        >>> result = parse_html(html)
        >>> print(result['title'])
        'Getting Started with Python'
        >>> print(len(result['messages']))
        12
    """
    parser = ParserFactory.create_parser(html_content)
    
    if parser:
        return parser.parse()
    
    return None


def parse_all_html(html_content: str) -> List[Dict]:
    """
    Parse ALL conversations from an HTML export.

    ChatGPT exports can contain dozens of conversations in one file.
    This returns every conversation found.  For formats that only
    support a single conversation (e.g. Claude), returns a one-element
    list.

    Args:
        html_content: Raw HTML string

    Returns:
        List of parsed conversation dicts (may be empty).
    """
    fmt = ParserFactory.detect_format_type(html_content)

    if fmt == 'chatgpt':
        parser = ChatGPTParser(html_content)
        return parser.parse_all()

    # For other formats, fall back to single parse
    parser = ParserFactory.create_parser(html_content)
    if parser:
        result = parser.parse()
        if result and result.get('messages'):
            return [result]

    return []


def detect_format(html_content: str) -> Optional[str]:
    """
    Detect the format of an HTML chat export.
    
    Args:
        html_content: Raw HTML string
        
    Returns:
        Format type string ('chatgpt', 'claude') or None if unrecognized
        
    Example:
        >>> html = open('export.html').read()
        >>> format_type = detect_format(html)
        >>> print(format_type)
        'chatgpt'
    """
    return ParserFactory.detect_format_type(html_content)


__all__ = [
    'BaseParser',
    'ChatGPTParser',
    'ClaudeParser',
    'ParserFactory',
    'parse_html',
    'parse_all_html',
    'detect_format',
]
