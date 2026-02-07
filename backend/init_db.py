#!/usr/bin/env python3
"""
Database initialization script for Cortex.

This script creates all database tables defined in models.py.
Run this script before starting the application for the first time.

Usage:
    python backend/init_db.py
    
Options:
    --drop    Drop all existing tables before creating new ones (WARNING: destroys data!)
    --seed    Add sample data for testing
"""
import sys
import argparse
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir.parent))

from backend.database import Base, engine, get_db_context, init_db, drop_db
from backend.models import Conversation, Message, Embedding, MessageRole
from datetime import datetime
import uuid


def create_sample_data():
    """Create sample conversations for testing."""
    print("Creating sample data...")
    
    with get_db_context() as db:
        # Sample conversation 1: Python Tutorial
        conv1 = Conversation(
            id=str(uuid.uuid4()),
            title="Getting Started with Python",
            summary="A beginner's guide to Python programming covering basics, data types, and control structures.",
            topics=["python", "programming", "tutorial", "basics"],
            cluster_id=0,
            cluster_name="Programming",
            message_count=4,
            created_at=datetime(2026, 1, 15, 10, 30, 0)
        )
        
        messages1 = [
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv1.id,
                role=MessageRole.USER,
                content="How do I get started with Python programming?",
                sequence_number=0,
                created_at=conv1.created_at
            ),
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv1.id,
                role=MessageRole.ASSISTANT,
                content="Python is a great language to start with! Here are the basics: 1. Install Python from python.org, 2. Learn basic syntax, 3. Practice with simple projects.",
                sequence_number=1,
                created_at=conv1.created_at
            ),
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv1.id,
                role=MessageRole.USER,
                content="What are the main data types in Python?",
                sequence_number=2,
                created_at=conv1.created_at
            ),
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv1.id,
                role=MessageRole.ASSISTANT,
                content="Python's main data types include: int, float, str, bool, list, tuple, dict, and set. Each has specific use cases.",
                sequence_number=3,
                created_at=conv1.created_at
            ),
        ]
        
        # Sample embedding for conversation 1
        embedding1 = Embedding(
            conversation_id=conv1.id,
            embedding_384d=[0.1] * 384,  # Placeholder embedding
            vector_3d=[1.5, 2.0, 1.0],
            start_x=0.0,
            start_y=0.0,
            start_z=0.0,
            end_x=1.5,
            end_y=2.0,
            end_z=1.0,
            magnitude=2.5
        )
        
        # Sample conversation 2: Machine Learning
        conv2 = Conversation(
            id=str(uuid.uuid4()),
            title="Introduction to Machine Learning",
            summary="Overview of machine learning concepts including supervised learning, neural networks, and model training.",
            topics=["machine learning", "AI", "neural networks", "training"],
            cluster_id=1,
            cluster_name="AI & ML",
            message_count=3,
            created_at=datetime(2026, 1, 20, 14, 45, 0)
        )
        
        messages2 = [
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv2.id,
                role=MessageRole.USER,
                content="What is machine learning?",
                sequence_number=0,
                created_at=conv2.created_at
            ),
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv2.id,
                role=MessageRole.ASSISTANT,
                content="Machine learning is a subset of AI that enables computers to learn from data without explicit programming. It uses algorithms to find patterns and make predictions.",
                sequence_number=1,
                created_at=conv2.created_at
            ),
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conv2.id,
                role=MessageRole.USER,
                content="How do neural networks work?",
                sequence_number=2,
                created_at=conv2.created_at
            ),
        ]
        
        embedding2 = Embedding(
            conversation_id=conv2.id,
            embedding_384d=[0.2] * 384,
            vector_3d=[-1.0, 1.5, 2.0],
            start_x=0.0,
            start_y=0.0,
            start_z=0.0,
            end_x=-1.0,
            end_y=1.5,
            end_z=2.0,
            magnitude=2.2
        )
        
        # Add all to database
        db.add(conv1)
        db.add_all(messages1)
        db.add(embedding1)
        
        db.add(conv2)
        db.add_all(messages2)
        db.add(embedding2)
        
        db.commit()
        
    print(f" Created 2 sample conversations with messages and embeddings")


def main():
    """Main function to initialize the database."""
    parser = argparse.ArgumentParser(description="Initialize Cortex database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables before creating")
    parser.add_argument("--seed", action="store_true", help="Add sample data for testing")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cortex Database Initialization")
    print("=" * 60)
    
    try:
        # Drop existing tables if requested
        if args.drop:
            print("")
            print("[WARNING] Dropping all existing tables")
            response = input("Are you sure? This will delete all data! (yes/no): ")
            if response.lower() == "yes":
                drop_db()
                print("Dropped all tables")
            else:
                print("Aborted")
                return

        # Create tables
        print("")
        print("Creating database tables")
        init_db()
        print("Successfully created all tables:")
        print("   - conversations")
        print("   - messages")
        print("   - embeddings")

        # Add sample data if requested
        if args.seed:
            print("")
            print("Seeding database with sample data")
            create_sample_data()

        print("")
        print("=" * 60)
        print("Database initialization complete!")
        print("=" * 60)
        print("")
        print("You can now start the backend server:")
        print("  uvicorn backend.main:app --reload")
        print("")

    except Exception as e:
        print("")
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()