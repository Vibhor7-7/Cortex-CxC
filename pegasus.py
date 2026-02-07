"""
Pegasus Model Test Script for Conversation Summarization

This script demonstrates using the Pegasus model for text summarization
as an alternative to the OpenAI GPT-4o-mini approach used in backend/services/summarizer.py.

Features:
- Handles large context windows (RAG pipeline style)
- Automatic chunking for texts >1024 tokens
- Generates short summaries (~50 words)
- Smart device detection (CUDA/MPS/CPU)
- Optimized for fast, scalable summarization

Note: The main project uses OpenAI for summarization. This is a standalone
test/demo script for exploring local summarization alternatives.
"""

import sys
import torch
from typing import Iterable, List

try:
    from transformers import pipeline
except ImportError:
    print("Error: transformers library not installed.")
    print("Install with: pip install transformers torch")
    sys.exit(1)


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0  # GPU device 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return "mps"
    else:
        print("Using CPU (this will be slower)")
        return -1  # CPU


def create_summarization_pipeline():
    """Create and configure the summarization pipeline."""
    device = get_device()

    try:
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer

        # Load tokenizer and model
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

        # Use float16 only for CUDA GPUs, float32 for CPU/MPS
        if device == 0:  # CUDA
            model = PegasusForConditionalGeneration.from_pretrained(
                "google/pegasus-xsum",
                torch_dtype=torch.float16
            ).to(device)
        else:  # CPU or MPS
            model = PegasusForConditionalGeneration.from_pretrained(
                "google/pegasus-xsum"
            )
            if device == "mps":
                model = model.to(device)
            # device -1 means CPU, no need to move

        model.eval()
        print("Model loaded successfully!\n")
        return model, tokenizer, device

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def _max_input_tokens(tokenizer, fallback=512):
    max_len = getattr(tokenizer, "model_max_length", fallback)
    if not isinstance(max_len, int) or max_len > 10000:
        max_len = fallback
    return max_len


def iter_chunks(text, tokenizer, max_chunk_tokens=None, overlap=80):
    """
    Yield overlapping chunks for processing large contexts without
    tokenizing the entire input at once.

    Args:
        text: Input text to chunk
        tokenizer: Tokenizer to use
        max_chunk_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        Generator of text chunks
    """
    if max_chunk_tokens is None:
        max_chunk_tokens = _max_input_tokens(tokenizer)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    current_tokens: List[int] = []
    current_text_parts: List[str] = []

    for para in paragraphs:
        para_tokens = tokenizer.encode(para, add_special_tokens=False)
        if len(para_tokens) > max_chunk_tokens:
            # Flush current chunk before splitting large paragraph
            if current_tokens:
                yield tokenizer.decode(current_tokens, skip_special_tokens=True)
                current_tokens = []
                current_text_parts = []

            start = 0
            while start < len(para_tokens):
                end = min(start + max_chunk_tokens, len(para_tokens))
                chunk_tokens = para_tokens[start:end]
                yield tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                start += max_chunk_tokens - overlap
            continue

        if len(current_tokens) + len(para_tokens) > max_chunk_tokens and current_tokens:
            yield tokenizer.decode(current_tokens, skip_special_tokens=True)
            if overlap > 0:
                overlap_tokens = current_tokens[-overlap:]
                current_tokens = overlap_tokens[:]
                current_text_parts = [tokenizer.decode(overlap_tokens, skip_special_tokens=True)]
            else:
                current_tokens = []
                current_text_parts = []

        current_tokens.extend(para_tokens)
        current_text_parts.append(para)

    if current_tokens:
        yield tokenizer.decode(current_tokens, skip_special_tokens=True)


def _to_device(inputs, device):
    if device == 0:
        return {k: v.to("cuda") for k, v in inputs.items()}
    if device == "mps":
        return {k: v.to("mps") for k, v in inputs.items()}
    return inputs


def _summarize_batch(texts, model, tokenizer, device, max_length, min_length):
    max_input = _max_input_tokens(tokenizer)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_input,
        truncation=True,
        padding=True
    )
    inputs = _to_device(inputs, device)
    with torch.inference_mode():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=1,  # fastest
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            repetition_penalty=1.4
        )
    return [tokenizer.decode(s, skip_special_tokens=True).strip() for s in summary_ids]


def summarize(text, model, tokenizer, device, target_words=70):
    """
    Summarize text using the Pegasus model.

    Designed for large context windows (RAG pipeline style) with short summaries.
    For very large texts (>1024 tokens), chunks the text and summarizes each chunk,
    then creates a final summary.

    Args:
        text: Input text to summarize (can be very long)
        model: Pegasus model
        tokenizer: Pegasus tokenizer
        device: Device (cuda/mps/cpu)
        target_words: Approximate target length in words (default 50)
    """
    # Rough token length heuristic for ~70 words.
    max_length = max(40, int(target_words * 1.5))
    min_length = max(22, int(target_words * 0.7))

    max_input = _max_input_tokens(tokenizer)

    # Fast path: short input
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count <= max_input:
        return _summarize_batch([text], model, tokenizer, device, max_length, min_length)[0]

    # Large input: chunk -> summarize -> reduce (iteratively)
    chunk_iter = iter_chunks(text, tokenizer, max_chunk_tokens=max_input, overlap=80)
    summaries: List[str] = []
    batch: List[str] = []
    batch_size = 6

    for chunk in chunk_iter:
        batch.append(chunk)
        if len(batch) >= batch_size:
            summaries.extend(_summarize_batch(batch, model, tokenizer, device, max_length, min_length))
            batch = []
            # Reduce periodically to keep memory small
            if len(summaries) >= 20:
                summaries = reduce_summaries(summaries, model, tokenizer, device, max_length, min_length)

    if batch:
        summaries.extend(_summarize_batch(batch, model, tokenizer, device, max_length, min_length))

    # Final reduce
    summaries = reduce_summaries(summaries, model, tokenizer, device, max_length, min_length)
    return summaries[0] if summaries else ""


def reduce_summaries(summaries, model, tokenizer, device, max_length, min_length):
    if len(summaries) <= 1:
        return summaries
    grouped: List[str] = []
    group_size = 6
    for i in range(0, len(summaries), group_size):
        grouped.append(" ".join(summaries[i:i+group_size]))
    return _summarize_batch(grouped, model, tokenizer, device, max_length, min_length)


def read_paragraph():
    """Read a paragraph from stdin (supports multi-line)."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Paste a paragraph to summarize. Finish with an empty line:\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    """Main function to run summarization."""
    model, tokenizer, device = create_summarization_pipeline()

    text = read_paragraph()
    if not text:
        print("No input provided.")
        return

    try:
        result = summarize(text, model, tokenizer, device, target_words=70)
        print(f"\nSummary (~70 words, {len(result.split())} words):")
        print(result)
    except Exception as e:
        print(f"Error during summarization: {e}")


if __name__ == "__main__":
    main()
