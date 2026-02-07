"""
Manual end-to-end backend workflow test.

Parses tests/chat.html → picks a random conversation → summarizes it → embeds it.
Run: python -m tests.manual_workflow_test
"""

import asyncio
import random
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.parsers import parse_html
from backend.parsers.chatgpt_parser import ChatGPTParser
from backend.services.summarizer import summarize_conversation
from backend.services.embedder import generate_embedding, prepare_text_for_embedding


def parse_all_conversations(html_content: str):
    """
    Parse ALL conversations from a ChatGPT HTML export (not just the first one).
    The default parser only returns conversations[0], so we dig into the JSON directly.
    """
    import json, re

    parser = ChatGPTParser(html_content)
    scripts = parser.soup.find_all('script')

    for script in scripts:
        script_text = script.string
        if not script_text:
            continue

        match = re.search(r'var\s+jsonData\s*=\s*\[', script_text)
        if match:
            start_pos = match.end() - 1
            json_str = parser._extract_json_array(script_text[start_pos:])
            if json_str:
                conversations_raw = json.loads(json_str)
                # Use the parser's own JSON-to-dict converter for each conversation
                return [
                    parser._parse_json_conversation(c) for c in conversations_raw
                ]

    # Fallback: single conversation via default parser
    result = parse_html(html_content)
    return [result] if result else []


async def main():
    # ── 1. Parse ──────────────────────────────────────────────────────
    html_path = os.path.join(os.path.dirname(__file__), "chat.html")
    print(f"📂 Reading {html_path} ...")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    conversations = parse_all_conversations(html_content)
    print(f"✅ Parsed {len(conversations)} conversations\n")

    if not conversations:
        print("❌ No conversations found!")
        return

    # List them all
    for i, conv in enumerate(conversations):
        n_msgs = len(conv.get("messages", []))
        print(f"   [{i:2d}] {conv['title']}  ({n_msgs} messages)")

    # ── 2. Pick a random conversation ─────────────────────────────────
    chosen = random.choice(conversations)
    print(f"\n🎲 Randomly selected: \"{chosen['title']}\"")
    msgs = chosen.get("messages", [])
    print(f"   Messages: {len(msgs)}")

    # Show first few messages as preview
    print("\n── Message preview (first 3) ──")
    for msg in msgs[:3]:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:200]
        print(f"   [{role}] {content}{'...' if len(msg.get('content','')) > 200 else ''}")

    # ── 3. Summarize ──────────────────────────────────────────────────
    print("\n⏳ Summarizing via Ollama (qwen2.5) ...")
    summary, topics = await summarize_conversation(
        messages=msgs,
        conversation_id=None,   # skip cache for this test
        use_cache=False
    )
    print(f"\n📝 Summary:\n   {summary}")
    print(f"\n🏷️  Topics: {topics}")

    # ── 4. Embed ──────────────────────────────────────────────────────
    embed_text = prepare_text_for_embedding(
        title=chosen["title"],
        summary=summary,
        topics=topics,
        messages=msgs
    )
    print(f"\n⏳ Generating embedding via Ollama (nomic-embed-text) ...")
    print(f"   Input text length: {len(embed_text)} chars")

    embedding = await generate_embedding(
        text=embed_text,
        conversation_id=None,   # skip cache
        use_cache=False
    )
    print(f"\n✅ Embedding generated!")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   First 10 values: {[round(v, 6) for v in embedding[:10]]}")
    print(f"   Min: {min(embedding):.6f}  Max: {max(embedding):.6f}")

    print("\n🎉 Full workflow complete: parse → summarize → embed")


if __name__ == "__main__":
    asyncio.run(main())
