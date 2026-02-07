"""
Base parser class for chat conversation HTML exports.

Provides common utilities for HTML parsing, text normalization,
and parser type detection.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re
from datetime import datetime


class BaseParser(ABC):
    """Abstract base class for chat conversation parsers."""
    
    def __init__(self, html_content: str):
        """
        Initialize parser with HTML content.
        
        Args:
            html_content: Raw HTML string of the conversation export
        """
        self.html_content = html_content
        self.soup = BeautifulSoup(html_content, 'lxml')
    
    @abstractmethod
    def parse(self) -> Dict:
        """
        Parse the HTML and extract conversation data.
        
        Returns:
            Dict with keys:
                - title: str
                - messages: List[Dict] with role, content, sequence_number
                - created_at: Optional[datetime]
        """
        pass
    
    @abstractmethod
    def detect_format(self) -> bool:
        """
        Detect if this parser can handle the given HTML format.
        
        Returns:
            True if this parser can handle the HTML, False otherwise
        """
        pass
    
    # ========================================================================
    # Common Utility Methods
    # ========================================================================
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text with normalized whitespace
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_code_blocks(self, element) -> List[Dict]:
        """
        Extract code blocks from HTML element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            List of dicts with 'language' and 'code' keys
        """
        code_blocks = []
        
        # Find all <pre> or <code> tags
        for code_tag in element.find_all(['pre', 'code']):
            language = code_tag.get('class', [''])[0] if code_tag.get('class') else 'text'
            code_text = code_tag.get_text()
            
            code_blocks.append({
                'language': language,
                'code': code_text
            })
        
        return code_blocks
    
    def normalize_role(self, role: str) -> str:
        """
        Normalize role names to standard format.
        
        Args:
            role: Raw role string
            
        Returns:
            Normalized role: 'user', 'assistant', or 'system'
        """
        role = role.lower().strip()
        
        # Map common variations
        role_mapping = {
            'user': 'user',
            'human': 'user',
            'you': 'user',
            'assistant': 'assistant',
            'ai': 'assistant',
            'chatgpt': 'assistant',
            'gpt': 'assistant',
            'claude': 'assistant',
            'system': 'system',
        }
        
        return role_mapping.get(role, 'assistant')
    
    def generate_title_from_first_message(self, messages: List[Dict], max_length: int = 50) -> str:
        """
        Generate a title from the first user message.
        
        Args:
            messages: List of message dicts
            max_length: Maximum length of the title
            
        Returns:
            Generated title string
        """
        for msg in messages:
            if msg.get('role') == 'user' and msg.get('content'):
                title = msg['content'][:max_length]
                if len(msg['content']) > max_length:
                    title += "..."
                return title
        
        return "Untitled Conversation"
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string to datetime object.
        
        Args:
            timestamp_str: Raw timestamp string
            
        Returns:
            datetime object or None if parsing fails
        """
        if not timestamp_str:
            return None
        
        # Common timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def extract_text_preserving_structure(self, element) -> str:
        """
        Extract text from HTML element while preserving some structure.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Text with preserved line breaks and structure
        """
        if not element:
            return ""
        
        # Replace <br> tags with newlines
        for br in element.find_all('br'):
            br.replace_with('\n')
        
        # Replace <p> tags with newlines
        for p in element.find_all('p'):
            p.insert_after('\n')
        
        # Get text and clean it
        text = element.get_text()
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return self.clean_text(text)
    
    def validate_messages(self, messages: List[Dict]) -> bool:
        """
        Validate that messages have required fields.
        
        Args:
            messages: List of message dicts
            
        Returns:
            True if all messages are valid, False otherwise
        """
        if not messages:
            return False
        
        required_fields = ['role', 'content', 'sequence_number']
        
        for msg in messages:
            if not all(field in msg for field in required_fields):
                return False
            
            if not msg['content'] or not msg['content'].strip():
                return False
        
        return True


class ParserFactory:
    """Factory class for creating appropriate parser based on HTML content."""
    
    @staticmethod
    def create_parser(html_content: str) -> Optional[BaseParser]:
        """
        Detect HTML format and create appropriate parser.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Instance of appropriate parser or None if format not recognized
        """
        # Import here to avoid circular imports
        from backend.parsers.chatgpt_parser import ChatGPTParser
        from backend.parsers.claude_parser import ClaudeParser
        
        # Try each parser's detection method
        parsers = [ChatGPTParser, ClaudeParser]
        
        for parser_class in parsers:
            parser = parser_class(html_content)
            if parser.detect_format():
                return parser
        
        return None
    
    @staticmethod
    def detect_format_type(html_content: str) -> Optional[str]:
        """
        Detect the format type without creating a parser.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            Format type string ('chatgpt', 'claude') or None
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Check title tag first
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().lower()
            if 'chatgpt' in title_text:
                return 'chatgpt'
            if 'claude' in title_text:
                return 'claude'
        
        # Check for ChatGPT indicators
        if soup.find(attrs={'class': re.compile(r'conversation', re.I)}):
            return 'chatgpt'
        
        # Check for Claude indicators
        if soup.find(attrs={'data-testid': re.compile(r'conversation', re.I)}):
            return 'claude'
        
        # Check for common OpenAI/Anthropic metadata
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            content = str(meta).lower()
            if 'openai' in content or 'chatgpt' in content:
                return 'chatgpt'
            if 'anthropic' in content or 'claude' in content:
                return 'claude'
        
        return None
