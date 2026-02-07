"""
Pegasus Model Test Script for Conversation Summarization

This script demonstrates using the Pegasus model for text summarization
as an alternative to the OpenAI GPT-4o-mini approach used in backend/services/summarizer.py.

Features:
- Handles large context windows (RAG pipeline style)
- Automatic chunking for texts >1024 tokens
- Generates detailed summaries (100-256 tokens)
- Smart device detection (CUDA/MPS/CPU)
- Optimized for long-form conversation summarization

Note: The main project uses OpenAI for summarization. This is a standalone
test/demo script for exploring local summarization alternatives.
"""

import sys
import torch

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

        print("Model loaded successfully!\n")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def chunk_text(text, tokenizer, max_chunk_tokens=1024, overlap=100):
    """
    Split text into overlapping chunks for processing large contexts.

    Args:
        text: Input text to chunk
        tokenizer: Tokenizer to use
        max_chunk_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_chunk_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_chunk_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_chunk_tokens - overlap

    return chunks


def summarize(text, model, tokenizer, device, max_length=256, min_length=100):
    """
    Summarize text using the Pegasus model.

    Designed for large context windows (RAG pipeline style) with detailed summaries.
    For very large texts (>1024 tokens), chunks the text and summarizes each chunk,
    then creates a final summary.

    Args:
        text: Input text to summarize (can be very long)
        model: Pegasus model
        tokenizer: Pegasus tokenizer
        device: Device (cuda/mps/cpu)
        max_length: Maximum summary length in tokens (default 256 for detailed summaries)
        min_length: Minimum summary length in tokens (default 100)
    """
    # Check if text needs chunking
    token_count = len(tokenizer.encode(text, add_special_tokens=False))

    if token_count > 1024:
        # For very large contexts, chunk and summarize progressively
        chunks = chunk_text(text, tokenizer, max_chunk_tokens=1024, overlap=100)
        print(f"  [Large context detected: {token_count} tokens split into {len(chunks)} chunks]")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)

            # Move to device
            if device == 0:  # CUDA
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            elif device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=128,  # Shorter for chunk summaries
                    min_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )

            chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(chunk_summary)

        # Combine chunk summaries and create final summary
        combined = " ".join(chunk_summaries)
        inputs = tokenizer(combined, return_tensors="pt", max_length=1024, truncation=True)
    else:
        # Normal processing for texts within token limit
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # Move to device
    if device == 0:  # CUDA
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    elif device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    # Generate summary with parameters optimized for longer, detailed summaries
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=6,  # More beams for better quality on longer summaries
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,  # Neutral length penalty for balanced summaries
            repetition_penalty=2.0,  # Avoid repetition in longer summaries
            temperature=1.0  # Standard temperature for coherent output
        )

    # Decode
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    """Main function to run summarization demo."""

    # Create the summarization pipeline
    model, tokenizer = create_summarization_pipeline()
    device = get_device()

    # Sample text to summarize (climate change example)
    text = """Climate change is one of the most pressing global
         challenges of the twenty-first century, affecting
         ecosystems, economies, and human health across the world.
         Rising global temperatures have led to more frequent and
         severe weather events such as heatwaves, floods, droughts,
         and hurricanes, placing strain on infrastructure and food
         systems. Human activities, particularly the burning of fossil
         fuels and large-scale deforestation, are the primary drivers of
         greenhouse gas emissions that accelerate this process. While international
         agreements and national policies aim to reduce emissions and promote renewable energy,
         progress remains uneven. Addressing climate change will require coordinated global action, technological innovation,
         and long-term changes in consumption and production patterns."""

    # Example conversation format (for AI chat summarization)
    conversation = """USER: How do I implement a binary search tree in Python?

ASSISTANT: I'll help you implement a binary search tree (BST) in Python. Here's a complete implementation with insertion, search, and traversal methods.

USER: Can you also add a delete method?

ASSISTANT: Absolutely! I'll add a delete method that handles three cases: deleting a leaf node, a node with one child, and a node with two children."""

    print("EXAMPLE 1: Summarizing Article Text")
    print(f"Original text ({len(text.split())} words):")
    print(text.strip()[:200] + "...")

    try:
        result = summarize(text, model, tokenizer, device, max_length=256, min_length=100)
        print(f"Summary ({len(result.split())} words):")
        print(result)
    except Exception as e:
        print(f"Error during summarization: {e}")

    print("EXAMPLE 2: Summarizing AI Conversation")
    print(f"\nOriginal conversation ({len(conversation.split())} words):")
    print(conversation)

    try:
        result2 = summarize(conversation, model, tokenizer, device, max_length=200, min_length=80)
        print(f"Summary ({len(result2.split())} words):")
        print(result2)
    except Exception as e:
        print(f"Error during summarization: {e}")

    # Example 3: Large RAG-style context (simulating retrieved documents)
    large_context = """USER: What are the best practices for implementing a scalable microservices architecture?

ASSISTANT: I'll provide comprehensive guidance on microservices best practices.

1. Service Design Principles:
- Single Responsibility: Each service should handle one business capability
- Bounded Context: Define clear boundaries using Domain-Driven Design
- API-First Design: Design APIs before implementation
- Loose Coupling: Minimize dependencies between services
- High Cohesion: Keep related functionality together

2. Communication Patterns:
- Synchronous: REST APIs, gRPC for real-time requests
- Asynchronous: Message queues (RabbitMQ, Kafka) for event-driven architecture
- Service Mesh: Implement service discovery and load balancing (Istio, Linkerd)

3. Data Management:
- Database per Service: Each service owns its data
- Event Sourcing: Store state changes as events
- CQRS: Separate read and write operations
- Saga Pattern: Handle distributed transactions

4. Deployment and Operations:
- Containerization: Use Docker for consistency
- Orchestration: Kubernetes for scaling and management
- CI/CD: Automated testing and deployment pipelines
- Monitoring: Distributed tracing (Jaeger), metrics (Prometheus), logging (ELK)

5. Security:
- API Gateway: Centralized authentication and authorization
- OAuth2/JWT: Secure service-to-service communication
- Secrets Management: Use Vault or cloud-native solutions
- Network Policies: Implement zero-trust networking

6. Resilience:
- Circuit Breakers: Prevent cascade failures
- Retry Logic: Handle transient failures
- Rate Limiting: Protect against overload
- Bulkheads: Isolate critical resources

USER: Can you elaborate on the saga pattern for distributed transactions?

ASSISTANT: Absolutely! The Saga pattern is crucial for maintaining data consistency across microservices.

A saga is a sequence of local transactions where each service performs its transaction and publishes an event. If any step fails, compensating transactions are executed to undo the changes.

Types of Sagas:
1. Choreography: Services listen to events and decide what to do next
2. Orchestration: A central coordinator manages the transaction flow

Example - E-commerce Order:
- Order Service: Create order (reserved state)
- Payment Service: Process payment
- Inventory Service: Reserve items
- Shipping Service: Schedule delivery

If payment fails, compensating transactions:
- Cancel order reservation
- Release inventory
- Notify customer

Implementation with event sourcing ensures you can replay events and maintain consistency even in failure scenarios."""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Large RAG-Style Context (Microservices Discussion)")
    print("=" * 70)
    print(f"\nOriginal context ({len(large_context.split())} words):")
    print(large_context[:300] + "...")

    try:
        result3 = summarize(large_context, model, tokenizer, device, max_length=256, min_length=100)
        print(f"\nSummary ({len(result3.split())} words):")
        print(result3)
    except Exception as e:
        print(f"Error during summarization: {e}")


if __name__ == "__main__":
    main()
