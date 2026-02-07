"""
ChatGPT HTML export parser.

Parses ChatGPT conversation exports and extracts structured data.
"""
from typing import Dict, List, Optional
from datetime import datetime
import re
import json
from backend.parsers.base_parser import BaseParser


class ChatGPTParser(BaseParser):
    """Parser for ChatGPT HTML conversation exports."""
    
    def detect_format(self) -> bool:
        """
        Detect if this is a ChatGPT export.
        
        Returns:
            True if ChatGPT format detected, False otherwise
        """
        # Look for ChatGPT-specific elements
        indicators = [
            self.soup.find('title', string=re.compile(r'ChatGPT', re.I)),
            self.soup.find(attrs={'class': re.compile(r'conversation', re.I)}),
            self.soup.find(attrs={'data-message-author-role': True}),
            self.soup.find(attrs={'class': re.compile(r'chatgpt', re.I)}),
        ]
        
        # Check meta tags for OpenAI/ChatGPT
        meta_tags = self.soup.find_all('meta')
        for meta in meta_tags:
            if meta.get('content') and 'openai' in str(meta.get('content')).lower():
                return True
        
        return any(indicators)
    
    def parse(self) -> Dict:
        """
        Parse ChatGPT HTML export.
        
        Returns:
            Dict with:
                - title: str
                - messages: List[Dict] with role, content, sequence_number
                - created_at: Optional[datetime]
        """
        # First try to extract from embedded JSON (newer ChatGPT exports)
        json_result = self._try_parse_json_data()
        if json_result and json_result['messages']:
            return json_result
        
        # Fallback to HTML parsing (older exports or if JSON parsing fails)
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
    
    def _try_parse_json_data(self) -> Optional[Dict]:
        """
        Try to extract conversation data from embedded JSON in script tags.
        This handles newer ChatGPT exports that embed JSON data.
        
        Returns:
            Dict with parsed data or None if no JSON found
        """
        # Find all script tags
        scripts = self.soup.find_all('script')
        
        for script in scripts:
            script_text = script.string
            if not script_text:
                continue
            
            # Look for JSON data assignments
            patterns = [
                r'var\s+jsonData\s*=\s*',
                r'const\s+conversations\s*=\s*',
                r'var\s+conversations\s*=\s*',
            ]
            
            for pattern in patterns:
                match = re.search(pattern + r'\[', script_text)
                if match:
                    # Found start of JSON array
                    start_pos = match.end() - 1  # Position of '['
                    json_text = script_text[start_pos:]
                    
                    # Extract the complete JSON array by counting brackets
                    json_str = self._extract_json_array(json_text)
                    
                    if json_str:
                        try:
                            conversations = json.loads(json_str)
                            
                            if conversations and isinstance(conversations, list):
                                # Process first conversation (or all if multiple)
                                return self._parse_json_conversation(conversations[0])
                                
                        except json.JSONDecodeError:
                            continue
        
        return None
    
    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        Extract a complete JSON array by counting brackets.
        
        Args:
            text: Text starting with '['
            
        Returns:
            Complete JSON string or None
        """
        bracket_count = 0
        end_pos = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '[' or char == '{':
                    bracket_count += 1
                elif char == ']' or char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break
        
        if end_pos > 0:
            return text[:end_pos]
        
        return None
    
    def _parse_json_conversation(self, conversation_data: Dict) -> Dict:
        """
        Parse a conversation from JSON data.
        
        Args:
            conversation_data: Dict with conversation JSON data
            
        Returns:
            Dict with parsed conversation
        """
        title = conversation_data.get('title', 'Untitled Conversation')
        
        # Extract timestamp
        created_at = None
        if 'create_time' in conversation_data:
            try:
                created_at = datetime.fromtimestamp(conversation_data['create_time'])
            except (ValueError, TypeError):
                pass
        
        # Extract messages from mapping
        messages = []
        mapping = conversation_data.get('mapping', {})
        
        # Build message tree and extract in order
        if mapping:
            # Find root node
            root_id = None
            for node_id, node in mapping.items():
                if node.get('parent') is None or node_id == 'client-created-root':
                    root_id = node_id
                    break
            
            # Traverse tree to get messages in order
            if root_id:
                messages = self._traverse_message_tree(mapping, root_id)
        
        # If no messages found via tree traversal, try alternative extraction
        if not messages:
            messages = self._extract_messages_flat(mapping)
        
        return {
            'title': title,
            'messages': messages,
            'created_at': created_at
        }
    
    def _traverse_message_tree(self, mapping: Dict, node_id: str, sequence: int = 0) -> List[Dict]:
        """
        Recursively traverse message tree to extract messages in order.
        
        Args:
            mapping: Message mapping dict
            node_id: Current node ID
            sequence: Current sequence number
            
        Returns:
            List of message dicts
        """
        messages = []
        
        node = mapping.get(node_id)
        if not node:
            return messages
        
        # Extract message from current node
        message_data = node.get('message')
        if message_data:
            author = message_data.get('author', {})
            role = author.get('role', 'assistant')
            role = self.normalize_role(role)
            
            # Extract content from message parts
            content_parts = []
            content_obj = message_data.get('content')
            
            if content_obj:
                if isinstance(content_obj, dict):
                    parts = content_obj.get('parts', [])
                    for part in parts:
                        if isinstance(part, str):
                            content_parts.append(part)
                        elif isinstance(part, dict):
                            # Handle structured parts
                            if 'text' in part:
                                content_parts.append(part['text'])
                elif isinstance(content_obj, str):
                    content_parts.append(content_obj)
            
            content = '\n'.join(content_parts).strip()
            
            # Only add if there's actual content and not a system message
            if content and role != 'system':
                messages.append({
                    'role': role,
                    'content': content,
                    'sequence_number': sequence + len(messages)
                })
        
        # Process children
        children = node.get('children', [])
        for child_id in children:
            child_messages = self._traverse_message_tree(mapping, child_id, sequence + len(messages))
            messages.extend(child_messages)
        
        return messages
    
    def _extract_messages_flat(self, mapping: Dict) -> List[Dict]:
        """
        Extract messages from mapping without tree traversal (fallback method).
        
        Args:
            mapping: Message mapping dict
            
        Returns:
            List of message dicts
        """
        messages = []
        
        for node_id, node in mapping.items():
            message_data = node.get('message')
            if not message_data:
                continue
            
            author = message_data.get('author', {})
            role = author.get('role', 'assistant')
            role = self.normalize_role(role)
            
            # Skip system messages
            if role == 'system':
                continue
            
            # Extract content
            content_parts = []
            content_obj = message_data.get('content')
            
            if content_obj:
                if isinstance(content_obj, dict):
                    parts = content_obj.get('parts', [])
                    for part in parts:
                        if isinstance(part, str):
                            content_parts.append(part)
                elif isinstance(content_obj, str):
                    content_parts.append(content_obj)
            
            content = '\n'.join(content_parts).strip()
            
            if content:
                messages.append({
                    'role': role,
                    'content': content,
                    'create_time': message_data.get('create_time', 0)
                })
        
        # Sort by create_time and assign sequence numbers
        messages.sort(key=lambda x: x.get('create_time', 0))
        for i, msg in enumerate(messages):
            msg['sequence_number'] = i
            msg.pop('create_time', None)
        
        return messages
    
    def _extract_title(self) -> str:
        """Extract conversation title from HTML."""
        # Try various selectors for title
        
        # Method 1: Look for title in <title> tag
        title_tag = self.soup.find('title')
        if title_tag:
            title = title_tag.get_text()
            # Remove "ChatGPT - " prefix if present
            title = re.sub(r'^ChatGPT\s*-\s*', '', title, flags=re.I)
            if title and title.strip() and title.lower() != 'chatgpt':
                return self.clean_text(title)
        
        # Method 2: Look for h1 or header with conversation title
        header = self.soup.find(['h1', 'h2'], attrs={'class': re.compile(r'title|heading', re.I)})
        if header:
            title = header.get_text()
            if title and title.strip():
                return self.clean_text(title)
        
        # Method 3: Look for specific data attributes
        title_elem = self.soup.find(attrs={'data-conversation-title': True})
        if title_elem:
            return self.clean_text(title_elem.get('data-conversation-title'))
        
        # Method 4: Look for first heading in main content
        main_content = self.soup.find(['main', 'article', 'div'], attrs={'class': re.compile(r'conversation|content|main', re.I)})
        if main_content:
            heading = main_content.find(['h1', 'h2', 'h3'])
            if heading:
                return self.clean_text(heading.get_text())
        
        return "Untitled Conversation"
    
    def _extract_timestamp(self) -> Optional[datetime]:
        """Extract conversation timestamp."""
        # Look for timestamp in various locations
        
        # Method 1: Meta tag with date
        meta_date = self.soup.find('meta', attrs={'name': re.compile(r'date|created', re.I)})
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
        
        # Method 3: Look for date in header or footer
        for elem in self.soup.find_all(['header', 'footer', 'div'], attrs={'class': re.compile(r'date|time|timestamp', re.I)}):
            timestamp = self.parse_timestamp(elem.get_text())
            if timestamp:
                return timestamp
        
        return None
    
    def _extract_messages(self) -> List[Dict]:
        """Extract all messages from the conversation."""
        messages = []
        sequence_number = 0
        
        # Method 1: Look for messages with data attributes (newer exports)
        message_elements = self.soup.find_all(attrs={'data-message-author-role': True})
        
        if message_elements:
            for elem in message_elements:
                role = elem.get('data-message-author-role', 'assistant')
                role = self.normalize_role(role)
                
                # Extract content
                content = self._extract_message_content(elem)
                
                if content and content.strip():
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        # Method 2: Look for alternating user/assistant divs (older exports)
        if not messages:
            message_elements = self.soup.find_all(['div', 'article'], attrs={'class': re.compile(r'message|chat-message|conversation-turn', re.I)})
            
            for elem in message_elements:
                # Try to determine role from class names or content
                classes = ' '.join(elem.get('class', [])).lower()
                
                if 'user' in classes or 'human' in classes:
                    role = 'user'
                elif 'assistant' in classes or 'ai' in classes or 'gpt' in classes:
                    role = 'assistant'
                else:
                    # Alternate between user and assistant
                    role = 'user' if sequence_number % 2 == 0 else 'assistant'
                
                content = self._extract_message_content(elem)
                
                if content and content.strip():
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        # Method 3: Fallback - look for any text content in paragraphs
        if not messages:
            paragraphs = self.soup.find_all('p')
            
            for i, p in enumerate(paragraphs):
                content = self.clean_text(p.get_text())
                
                if content and len(content) > 10:  # Ignore very short paragraphs
                    role = 'user' if i % 2 == 0 else 'assistant'
                    messages.append({
                        'role': role,
                        'content': content,
                        'sequence_number': sequence_number
                    })
                    sequence_number += 1
        
        return messages
    
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
            element.find(attrs={'class': re.compile(r'content|text|body', re.I)}) or
            element.find(['div', 'p', 'span'], recursive=False) or
            element
        )
        
        # Extract text while preserving structure
        content = self.extract_text_preserving_structure(content_elem)
        
        # Handle code blocks specially
        code_blocks = self.extract_code_blocks(content_elem)
        
        # If there are code blocks, format them nicely
        if code_blocks:
            for code_block in code_blocks:
                # Add code block markers
                code_content = f"\n```{code_block['language']}\n{code_block['code']}\n```\n"
                # This is a simplified approach - in real parsing we'd need to track positions
                content += code_content
        
        return self.clean_text(content)
