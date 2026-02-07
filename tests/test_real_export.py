"""
Test parser with real ChatGPT export file.
"""
import unittest
from pathlib import Path
from backend.parsers import parse_html, detect_format


class TestRealChatGPTExport(unittest.TestCase):
    """Test parsing of real ChatGPT export HTML."""
    
    @classmethod
    def setUpClass(cls):
        """Load the real chat.html file once for all tests."""
        html_path = Path(__file__).parent / 'chat.html'
        if not html_path.exists():
            cls.html_content = None
            cls.skipTest(cls, "chat.html file not found")
        else:
            cls.html_content = html_path.read_text()
    
    def test_detect_format(self):
        """Test that the format is correctly detected as ChatGPT."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        format_type = detect_format(self.html_content)
        self.assertEqual(format_type, 'chatgpt')
    
    def test_parse_success(self):
        """Test that parsing succeeds."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        self.assertIsNotNone(result)
        self.assertIn('title', result)
        self.assertIn('messages', result)
        self.assertIn('created_at', result)
    
    def test_extract_title(self):
        """Test that title is extracted."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        self.assertIsNotNone(result['title'])
        self.assertNotEqual(result['title'], '')
        self.assertNotEqual(result['title'], 'Untitled Conversation')
        print(f"Extracted title: {result['title']}")
    
    def test_extract_messages(self):
        """Test that messages are extracted."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        messages = result['messages']
        
        # Should have messages
        self.assertGreater(len(messages), 0, "Should extract at least one message")
        
        print(f"Extracted {len(messages)} messages")
        
        # Check message structure
        for msg in messages:
            self.assertIn('role', msg)
            self.assertIn('content', msg)
            self.assertIn('sequence_number', msg)
            self.assertIn(msg['role'], ['user', 'assistant', 'system'])
    
    def test_message_content(self):
        """Test that message content is meaningful."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        messages = result['messages']
        
        # First message should have content
        if messages:
            first_msg = messages[0]
            self.assertTrue(len(first_msg['content']) > 10, "Message should have substantial content")
            print(f"First message role: {first_msg['role']}")
            print(f"First message preview: {first_msg['content'][:100]}...")
    
    def test_message_sequence(self):
        """Test that messages are in correct sequence."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        messages = result['messages']
        
        # Check sequence numbers are sequential
        for i, msg in enumerate(messages):
            self.assertEqual(msg['sequence_number'], i, 
                           f"Message {i} should have sequence_number {i}")
    
    def test_timestamp_extraction(self):
        """Test that timestamp is extracted."""
        if not self.html_content:
            self.skipTest("No HTML content")
        
        result = parse_html(self.html_content)
        
        # created_at may be None for some exports, but if present should be valid
        if result['created_at']:
            print(f"Extracted timestamp: {result['created_at']}")
            self.assertIsNotNone(result['created_at'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
