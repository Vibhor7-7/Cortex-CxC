"""
Claude HTML export parser.

Parses Claude (Anthropic) conversation exports and extracts structured data.
"""
from typing import Dict, List, Optional
from datetime import datetime
import re
from backend.parsers.base_parser import BaseParser


class ClaudeParser(BaseParser):
    """Parser for Claude (Anthropic) HTML conversation exports."""
    
    def detect_format(self) -> bool:
        """
        Detect if this is a Claude export.
        
        Returns:
            True if Claude format detected, False otherwise
        """
        # Look for Claude-specific elements
        indicators = [
            self.soup.find('title', string=re.compile(r'Claude', re.I)),
            self.soup.find(attrs={'data-testid': re.compile(r'conversation|message', re.I)}),
            self.soup.find(attrs={'class': re.compile(r'claude', re.I)}),
            self.soup.find(attrs={'data-component': re.compile(r'message|chat', re.I)}),
        ]
        
        # Check meta tags for Anthropic/Claude
        meta_tags = self.soup.find_all('meta')
        for meta in meta_tags:
            content = str(meta.get('content', '')).lower()
            if 'anthropic' in content or 'claude' in content:
                return True
        
        # Check for Claude-specific attributes
        if self.soup.find(attrs={'data-claude-message': True}):
            return True
        
        return any(indicators)
    
    def parse(self) -> Dict:
        """
        Parse Claude HTML export.
        
        Returns:
            Dict with:
                - title: str
                - messages: List[Dict] with role, content, sequence_number
                - created_at: Optional[datetime]
        """
        # Extract title
        title = self._extract_title()
        
        # Extract messages
        messages = self._extract_messages()
        
        # Extract timestamp
        created_at = self._extract_timestamp()
        
        # Generate title from first message if not found
        if not title or title == "Untitled Conversation":
            title = self.generate_title_from_first_message(messages)
        
        return {
            'title': title,
            'messages': messages,
            'created_at': created_at
        }
    
    def _extract_title(self) -> str:
        """Extract conversation title from HTML."""
        # Try various selectors for title
        
        # Method 1: Look for title in <title> tag
        title_tag = self.soup.find('title')
        if title_tag:
            title = title_tag.get_text()
            # Remove "Claude - " or "Conversation with Claude" prefix if present
            title = re.sub(r'^(Claude\s*-\s*|Conversation\s+with\s+Claude\s*-?\s*)', '', title, flags=re.I)
            if title and title.strip() and title.lower() not in ['claude', 'anthropic']:
                return self.clean_text(title)
        
        # Method 2: Look for h1 or header with conversation title
        header = self.soup.find(['h1', 'h2'], attrs={'class': re.compile(r'title|heading|conversation-title', re.I)})
        if header:
            title = header.get_text()
            if title and title.strip():
                return self.clean_text(title)
        
        # Method 3: Look for specific data attributes
        title_elem = self.soup.find(attrs={'data-conversation-title': True})
        if title_elem:
            return self.clean_text(title_elem.get('data-conversation-title'))
        
        # Method 4: Look for title in header section
        header_section = self.soup.find(['header', 'div'], attrs={'class': re.compile(r'header|top', re.I)})
        if header_section:
            heading = header_section.find(['h1', 'h2', 'h3'])
            if heading:
                return self.clean_text(heading.get_text())
        
        return "Untitled Conversation"
    
    def _extract_timestamp(self) -> Optional[datetime]:
        """Extract conversation timestamp."""
        # Look for timestamp in various locations
        
        # Method 1: Meta tag with date
        meta_date = self.soup.find('meta', attrs={'name': re.compile(r'date|created|published', re.I)})
        if meta_date and meta_date.get('content'):
            timestamp = self.parse_timestamp(meta_date.get('content'))
            if timestamp:
                return timestamp
        
        # Method 2: Time tag
        time_tag = self.soup.find('time')
        if time_tag:
            if time_tag.get('datetime'):
                timestamp = self.parse_timestamp(time_tag.get('datetime'))
                if timestamp:
                    return timestamp
            timestamp = self.parse_timestamp(time_tag.get_text())
            if timestamp:
                return timestamp
        
        # Method 3: Look for date in specific data attributes
        date_elem = self.soup.find(attrs={'data-timestamp': True})
        if date_elem:
            timestamp = self.parse_timestamp(date_elem.get('data-timestamp'))
            if timestamp:
                return timestamp
        
        # Method 4: Look for date in header or footer
        for elem in self.soup.find_all(['header', 'footer', 'div'], attrs={'class': re.compile(r'date|time|timestamp', re.I)}):
            timestamp = self.parse_timestamp(elem.get_text())
            if timestamp:
                return timestamp
        
        return None
    
    def _extract_messages(self) -> List[Dict]:
        """Extract all messages from the conversation."""
        messages = []
        sequence_number = 0
        
        # Method 1: Look for messages with data-testid attributes (React exports)
        message_elements = self.soup.find_all(attrs={'data-testid': re.compile(r'message|conversation-turn', re.I)})
        
        if message_elements:
            for elem in message_elements:
                role = self._determine_role_from_element(elem)
                content = self._extract_message_content(elem)
                
                if content and content.strip():
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        # Method 2: Look for messages with specific class patterns
        if not messages:
            message_elements = self.soup.find_all(['div', 'article'], attrs={'class': re.compile(r'message|chat-message|turn', re.I)})
            
            for elem in message_elements:
                role = self._determine_role_from_element(elem)
                content = self._extract_message_content(elem)
                
                if content and content.strip():
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        # Method 3: Look for alternating message blocks
        if not messages:
            # Find main conversation container
            conversation = self.soup.find(['main', 'div'], attrs={'class': re.compile(r'conversation|messages|chat', re.I)})
            
            if conversation:
                # Find direct children that might be messages
                for child in conversation.find_all(['div', 'article'], recursive=False):
                    content = self._extract_message_content(child)
                    
                    if content and len(content) > 10:  # Ignore very short content
                        # Alternate roles
                        role = 'user' if sequence_number % 2 == 0 else 'assistant'
                        messages.append({
                            'role': role,
                            'content': content,
                            'sequence_number': sequence_number
                        })
                        sequence_number += 1
        
        # Method 4: Fallback - look for paragraphs
        if not messages:
            paragraphs = self.soup.find_all('p')
            
            for i, p in enumerate(paragraphs):
                content = self.clean_text(p.get_text())
                
                if content and len(content) > 10:
                    role = 'user' if i % 2 == 0 else 'assistant'
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        return messages
    
    def _determine_role_from_element(self, element) -> str:
        """
        Determine the message role from HTML element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Role string: 'user' or 'assistant'
        """
        # Check data-testid attribute first (most reliable for Claude)
        testid = element.get('data-testid', '')
        if testid:
            testid_lower = testid.lower()
            if 'user' in testid_lower or 'human' in testid_lower:
                return 'user'
            if 'assistant' in testid_lower or 'claude' in testid_lower:
                return 'assistant'
        
        # Check data attributes
        if element.get('data-author-role'):
            return self.normalize_role(element.get('data-author-role'))
        
        if element.get('data-message-author'):
            return self.normalize_role(element.get('data-message-author'))
        
        # Check class names
        classes = ' '.join(element.get('class', [])).lower()
        
        if any(term in classes for term in ['user', 'human', 'you']):
            return 'user'
        
        if any(term in classes for term in ['assistant', 'ai', 'claude', 'bot']):
            return 'assistant'
        
        # Check for role indicators in child elements
        role_elem = element.find(attrs={'class': re.compile(r'role|author', re.I)})
        if role_elem:
            role_text = role_elem.get_text().lower()
            return self.normalize_role(role_text)
        
        # Default to assistant if unclear
        return 'assistant'
    
    def _extract_message_content(self, element) -> str:
        """
        Extract content from a message element.
        
        Args:
            element: BeautifulSoup element containing the message
            
        Returns:
            Cleaned message content
        """
        # Look for the main content area within the message
        content_elem = (
            element.find(attrs={'class': re.compile(r'content|text|body|prose', re.I)}) or
            element.find(attrs={'data-testid': re.compile(r'content|text', re.I)}) or
            element.find(['div', 'p'], recursive=False) or
            element
        )
        
        # Remove unwanted elements (buttons, metadata, etc.)
        for unwanted in content_elem.find_all(['button', 'svg', 'img']):
            unwanted.decompose()
        
        # Extract text while preserving structure
        content = self.extract_text_preserving_structure(content_elem)
        
        # Handle code blocks specially
        code_blocks = self.extract_code_blocks(content_elem)
        
        # If there are code blocks, format them nicely
        if code_blocks:
            for code_block in code_blocks:
                # Add code block markers
                code_content = f"\n```{code_block['language']}\n{code_block['code']}\n```\n"
                content += code_content
        
        return self.clean_text(content)
