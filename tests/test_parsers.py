"""
Unit tests for HTML chat parsers.

Tests ChatGPT and Claude parser functionality with sample HTML
and edge cases.
"""
import unittest
from datetime import datetime
from backend.parsers import parse_html, detect_format, ChatGPTParser, ClaudeParser


class TestChatGPTParser(unittest.TestCase):
    """Test ChatGPT parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_chatgpt_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChatGPT - Python Tutorial</title>
            <meta name="author" content="OpenAI">
        </head>
        <body>
            <div class="conversation">
                <div class="message" data-message-author-role="user">
                    <div class="content">
                        <p>How do I get started with Python?</p>
                    </div>
                </div>
                <div class="message" data-message-author-role="assistant">
                    <div class="content">
                        <p>Python is a great language to start with! Here are the steps:</p>
                        <ol>
                            <li>Install Python from python.org</li>
                            <li>Learn basic syntax</li>
                            <li>Practice with simple projects</li>
                        </ol>
                    </div>
                </div>
                <div class="message" data-message-author-role="user">
                    <div class="content">
                        <p>Can you show me a hello world example?</p>
                    </div>
                </div>
                <div class="message" data-message-author-role="assistant">
                    <div class="content">
                        <p>Sure! Here's a simple example:</p>
                        <pre><code class="python">print("Hello, World!")</code></pre>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def test_detect_chatgpt_format(self):
        """Test ChatGPT format detection."""
        parser = ChatGPTParser(self.sample_chatgpt_html)
        self.assertTrue(parser.detect_format())
    
    def test_parse_chatgpt_title(self):
        """Test title extraction from ChatGPT export."""
        parser = ChatGPTParser(self.sample_chatgpt_html)
        result = parser.parse()
        
        self.assertEqual(result['title'], 'Python Tutorial')
    
    def test_parse_chatgpt_messages(self):
        """Test message extraction from ChatGPT export."""
        parser = ChatGPTParser(self.sample_chatgpt_html)
        result = parser.parse()
        
        messages = result['messages']
        self.assertEqual(len(messages), 4)
        
        # Check first message
        self.assertEqual(messages[0]['role'], 'user')
        self.assertIn('get started with Python', messages[0]['content'])
        self.assertEqual(messages[0]['sequence_number'], 0)
        
        # Check second message
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertIn('great language', messages[1]['content'])
        self.assertEqual(messages[1]['sequence_number'], 1)
    
    def test_parse_chatgpt_code_blocks(self):
        """Test code block extraction."""
        parser = ChatGPTParser(self.sample_chatgpt_html)
        result = parser.parse()
        
        # Last message should contain code
        last_message = result['messages'][-1]
        self.assertIn('print', last_message['content'])


class TestClaudeParser(unittest.TestCase):
    """Test Claude parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_claude_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Claude - Machine Learning Discussion</title>
            <meta name="author" content="Anthropic">
        </head>
        <body>
            <div class="conversation">
                <article data-testid="user-message">
                    <div class="prose">
                        <p>What is machine learning?</p>
                    </div>
                </article>
                <article data-testid="assistant-message">
                    <div class="prose">
                        <p>Machine learning is a subset of AI that enables computers to learn from data without explicit programming.</p>
                    </div>
                </article>
                <article data-testid="user-message">
                    <div class="prose">
                        <p>Can you give me an example?</p>
                    </div>
                </article>
                <article data-testid="assistant-message">
                    <div class="prose">
                        <p>Sure! Email spam detection is a classic example of machine learning.</p>
                    </div>
                </article>
            </div>
        </body>
        </html>
        """
    
    def test_detect_claude_format(self):
        """Test Claude format detection."""
        parser = ClaudeParser(self.sample_claude_html)
        self.assertTrue(parser.detect_format())
    
    def test_parse_claude_title(self):
        """Test title extraction from Claude export."""
        parser = ClaudeParser(self.sample_claude_html)
        result = parser.parse()
        
        self.assertEqual(result['title'], 'Machine Learning Discussion')
    
    def test_parse_claude_messages(self):
        """Test message extraction from Claude export."""
        parser = ClaudeParser(self.sample_claude_html)
        result = parser.parse()
        
        messages = result['messages']
        self.assertEqual(len(messages), 4)
        
        # Check message roles alternate correctly
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[2]['role'], 'user')
        self.assertEqual(messages[3]['role'], 'assistant')
    
    def test_parse_claude_content(self):
        """Test content extraction from Claude export."""
        parser = ClaudeParser(self.sample_claude_html)
        result = parser.parse()
        
        # Check first message content
        self.assertIn('machine learning', result['messages'][0]['content'].lower())
        
        # Check assistant response
        self.assertIn('subset of AI', result['messages'][1]['content'])


class TestParserFactory(unittest.TestCase):
    """Test parser factory and auto-detection."""
    
    def test_detect_chatgpt_format(self):
        """Test auto-detection of ChatGPT format."""
        html = '<html><head><title>ChatGPT - Test</title></head><body></body></html>'
        format_type = detect_format(html)
        self.assertEqual(format_type, 'chatgpt')
    
    def test_detect_claude_format(self):
        """Test auto-detection of Claude format."""
        html = '<html><head><title>Claude - Test</title></head><body></body></html>'
        format_type = detect_format(html)
        self.assertEqual(format_type, 'claude')
    
    def test_parse_with_auto_detection(self):
        """Test parsing with auto-detection."""
        html = """
        <html>
        <head><title>ChatGPT - Test</title></head>
        <body>
            <div data-message-author-role="user">
                <p>Hello</p>
            </div>
        </body>
        </html>
        """
        result = parse_html(html)
        
        self.assertIsNotNone(result)
        self.assertIn('title', result)
        self.assertIn('messages', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_html(self):
        """Test parsing empty HTML."""
        result = parse_html("")
        self.assertIsNone(result)
    
    def test_malformed_html(self):
        """Test parsing malformed HTML."""
        html = "<html><div>Incomplete"
        # Should not raise an exception
        result = parse_html(html)
        # May return None or partial data
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_no_messages(self):
        """Test HTML with no messages."""
        html = "<html><head><title>ChatGPT</title></head><body></body></html>"
        parser = ChatGPTParser(html)
        result = parser.parse()
        
        # Should return empty messages list
        self.assertEqual(len(result['messages']), 0)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        html = """
        <html>
        <head><title>ChatGPT - Test</title></head>
        <body>
            <div data-message-author-role="user">
                <p>Test with special chars: &lt;script&gt; &amp; "quotes"</p>
            </div>
        </body>
        </html>
        """
        parser = ChatGPTParser(html)
        result = parser.parse()
        
        # Special characters should be decoded
        self.assertIn('special chars', result['messages'][0]['content'])
    
    def test_very_long_message(self):
        """Test handling of very long messages."""
        long_content = "A" * 10000
        html = f"""
        <html>
        <head><title>ChatGPT - Test</title></head>
        <body>
            <div data-message-author-role="user">
                <p>{long_content}</p>
            </div>
        </body>
        </html>
        """
        parser = ChatGPTParser(html)
        result = parser.parse()
        
        # Should handle long content without errors
        self.assertTrue(len(result['messages'][0]['content']) > 5000)


class TestTextNormalization(unittest.TestCase):
    """Test text cleaning and normalization utilities."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        from backend.parsers.base_parser import BaseParser
        
        # Create a simple parser instance for testing
        parser = ChatGPTParser("<html></html>")
        
        # Test whitespace normalization
        dirty_text = "Test   with    multiple     spaces"
        clean = parser.clean_text(dirty_text)
        self.assertEqual(clean, "Test with multiple spaces")
        
        # Test leading/trailing whitespace
        dirty_text = "  \n  Test  \n  "
        clean = parser.clean_text(dirty_text)
        self.assertEqual(clean, "Test")
    
    def test_role_normalization(self):
        """Test role name normalization."""
        from backend.parsers.base_parser import BaseParser
        
        parser = ChatGPTParser("<html></html>")
        
        # Test various role names
        self.assertEqual(parser.normalize_role("User"), "user")
        self.assertEqual(parser.normalize_role("HUMAN"), "user")
        self.assertEqual(parser.normalize_role("Assistant"), "assistant")
        self.assertEqual(parser.normalize_role("ChatGPT"), "assistant")
        self.assertEqual(parser.normalize_role("Claude"), "assistant")
        self.assertEqual(parser.normalize_role("System"), "system")


if __name__ == '__main__':
    unittest.main()
