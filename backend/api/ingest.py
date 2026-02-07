"""
Chat ingestion API endpoint.

Handles uploading and processing of chat HTML files.
"""

import asyncio
import time
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from sqlalchemy.orm import Session
from backend.database import get_db_context
from backend.models import Conversation, Message, Embedding, MessageRole
from backend.schemas import IngestResponse, IngestBatchResponse
from backend.parsers import parse_html, parse_all_html, detect_format
from backend.services.normalizer import normalize_conversation
from backend.services.summarizer import summarize_conversation
from backend.services.embedder import generate_embedding, prepare_text_for_embedding
from backend.services.dimensionality_reducer import fit_umap_model, reduce_embeddings, normalize_coordinates
from backend.services.clusterer import cluster_conversations
from backend.services.vector_store import upsert_conversation_to_store
import uuid


router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/", response_model=IngestResponse)
async def ingest_single_chat(
    file: UploadFile = File(..., description="HTML chat export file"),
    auto_reprocess: bool = Form(default=False, description="Automatically re-run UMAP and clustering after ingestion")
):
    """
    Ingest a single chat HTML file.

    The file may contain **multiple** conversations (e.g. a ChatGPT bulk
    export).  Every conversation found is ingested.  The response reports
    on the *last* conversation processed; use the ``/batch`` endpoint for
    richer per-file feedback.

    Args:
        file: HTML chat export file (ChatGPT or Claude format)
        auto_reprocess: If True, re-run UMAP/clustering after ingestion

    Returns:
        IngestResponse with conversation ID and metadata
    """
    start_time = time.time()

    try:
        # Validate file type
        if not file.filename.endswith('.html'):
            raise HTTPException(
                status_code=400,
                detail="Only HTML files are accepted"
            )

        # Read file content
        content = await file.read()
        html_content = content.decode('utf-8')

        # Detect format
        format_type = detect_format(html_content)
        if not format_type:
            raise HTTPException(
                status_code=422,
                detail="Unable to detect chat format (ChatGPT/Claude)"
            )

        # Parse ALL conversations from the file
        all_conversations = parse_all_html(html_content)
        if not all_conversations:
            raise HTTPException(
                status_code=422,
                detail="No conversations found in the uploaded file"
            )

        print(f"[ingest] Found {len(all_conversations)} conversation(s) in {file.filename}")

        last_id = None
        last_title = None
        total_messages = 0
        ingested = 0

        for conv_idx, parsed_data in enumerate(all_conversations):
            messages = parsed_data.get('messages', [])
            if not messages:
                print(f"  [skip] conversation #{conv_idx}: no messages")
                continue

            # Normalize
            normalized = normalize_conversation(parsed_data, messages)

            # Summarise
            try:
                summary, topics = await summarize_conversation(normalized['messages'])
            except Exception as e:
                print(f"  [warn] Summarization failed for conv #{conv_idx}: {e}")
                summary = f"Conversation with {normalized['message_count']} messages"
                topics = []

            conversation_id = str(uuid.uuid4())

            # Embedding
            embedding_text = prepare_text_for_embedding(
                title=normalized['title'],
                summary=summary,
                topics=topics,
                messages=normalized['messages']
            )
            try:
                embedding_vector = await generate_embedding(
                    embedding_text,
                    conversation_id=conversation_id
                )
            except Exception as e:
                print(f"  [error] Embedding failed for conv #{conv_idx} ({normalized['title']}): {e}")
                continue  # skip this conversation, don't fail the whole upload

            vector_3d = [0.0, 0.0, 0.0]

            # Persist to DB
            with get_db_context() as db:
                conversation = Conversation(
                    id=conversation_id,
                    title=normalized['title'],
                    summary=summary,
                    topics=topics,
                    cluster_id=0,
                    cluster_name="Unclustered",
                    message_count=normalized['message_count'],
                    created_at=normalized['created_at']
                )
                db.add(conversation)

                for msg in normalized['messages']:
                    message = Message(
                        id=str(uuid.uuid4()),
                        conversation_id=conversation_id,
                        role=MessageRole(msg['role']),
                        content=msg['content'],
                        sequence_number=msg['sequence_number']
                    )
                    db.add(message)

                embedding = Embedding(
                    conversation_id=conversation_id,
                    embedding_384d=embedding_vector,
                    vector_3d=vector_3d,
                    start_x=0.0, start_y=0.0, start_z=0.0,
                    end_x=0.0,   end_y=0.0,   end_z=0.0,
                    magnitude=1.0
                )
                db.add(embedding)
                db.commit()

            # Vector store
            try:
                conversation_data = {
                    'title': normalized['title'],
                    'summary': summary,
                    'topics': topics,
                    'messages': normalized['messages']
                }
                await upsert_conversation_to_store(
                    conversation_id=conversation_id,
                    conversation_data=conversation_data,
                    embedding=embedding_vector,
                )
            except Exception as e:
                print(f"  [warn] Vector store upsert failed: {e}")

            ingested += 1
            total_messages += normalized['message_count']
            last_id = conversation_id
            last_title = normalized['title']
            print(f"  [ok] #{conv_idx+1}/{len(all_conversations)}: {normalized['title']} ({normalized['message_count']} msgs)")

        if ingested == 0:
            raise HTTPException(status_code=422, detail="All conversations in the file were empty or failed processing")

        # Reprocess UMAP + clustering so nodes get real 3D coordinates
        if auto_reprocess or ingested > 1:
            try:
                print(f"[ingest] Auto-reprocessing {ingested} new conversations…")
                await reprocess_all_conversations()
                print("[ingest] Reprocessing complete")
            except Exception as e:
                print(f"[warn] Reprocessing failed (positions will be at origin): {e}")

        processing_time = (time.time() - start_time) * 1000

        return IngestResponse(
            success=True,
            conversation_id=last_id,
            title=f"{ingested} conversations" if ingested > 1 else last_title,
            message_count=total_messages,
            error=None,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return IngestResponse(
            success=False,
            conversation_id=None,
            title=None,
            message_count=0,
            error=str(e),
            processing_time_ms=processing_time
        )


@router.post("/batch", response_model=IngestBatchResponse)
async def ingest_batch_chats(
    files: List[UploadFile] = File(..., description="Multiple HTML chat export files"),
    auto_reprocess: bool = Form(default=True, description="Automatically re-run UMAP and clustering after ingestion")
):
    """
    Ingest multiple chat HTML files in batch.

    Processes files sequentially and returns results for each.
    After successful ingestion, optionally triggers re-clustering to update
    3D coordinates and cluster assignments for all conversations.

    Args:
        files: List of HTML files to ingest
        auto_reprocess: If True, automatically re-run UMAP/clustering after ingestion (default: True)

    Returns:
        IngestBatchResponse with results for all files
    """
    start_time = time.time()

    results = []
    successful = 0
    failed = 0

    for file in files:
        try:
            result = await ingest_single_chat(file)
            if result.success:
                successful += 1
            else:
                failed += 1
            results.append(result)
        except Exception as e:
            failed += 1
            results.append(IngestResponse(
                success=False,
                conversation_id=None,
                title=file.filename,
                message_count=0,
                error=str(e),
                processing_time_ms=0
            ))

    # Trigger automatic reprocessing if requested and we had successful ingestions
    if auto_reprocess and successful > 0:
        try:
            print("Auto-reprocessing: Running UMAP and clustering on all conversations")
            await reprocess_all_conversations()
            print("Auto-reprocessing completed successfully")
        except Exception as e:
            print(f"Warning: Auto-reprocessing failed: {e}")
            # Don't fail the entire batch if reprocessing fails

    total_time = (time.time() - start_time) * 1000

    return IngestBatchResponse(
        total_processed=len(files),
        successful=successful,
        failed=failed,
        conversations=results,
        total_time_ms=total_time
    )


@router.post("/reprocess")
async def reprocess_all_conversations():
    """
    Re-run UMAP and clustering on all conversations.

    This endpoint should be called after ingesting new conversations to:
    1. Load all conversation embeddings from database
    2. Run UMAP dimensionality reduction (768D → 3D)
    3. Run K-means clustering
    4. Update database with new 3D coordinates and cluster assignments

    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()

    try:
        with get_db_context() as db:
            # 1. Load all conversations and embeddings
            embeddings_data = db.query(Embedding).all()

            if len(embeddings_data) < 2:
                raise HTTPException(
                    status_code=422,
                    detail="Need at least 2 conversations to perform clustering"
                )

            # Extract embeddings and conversation IDs
            conversation_ids = [emb.conversation_id for emb in embeddings_data]
            embedding_vectors = [emb.embedding_384d for emb in embeddings_data]

            # Get all topics and titles for cluster naming
            conversations = db.query(Conversation).filter(
                Conversation.id.in_(conversation_ids)
            ).all()
            # Build a lookup so order matches conversation_ids
            conv_map = {c.id: c for c in conversations}
            all_topics = [conv_map[cid].topics if cid in conv_map else [] for cid in conversation_ids]
            all_titles = [conv_map[cid].title if cid in conv_map else "" for cid in conversation_ids]

            # 2. Run UMAP dimensionality reduction
            print(f"Running UMAP on {len(embedding_vectors)} conversations...")
            # Fit a UMAP model, then reduce
            umap_model = fit_umap_model(
                embedding_vectors,
                save_model=True
            )
            reduced_data = reduce_embeddings(
                embedding_vectors,
                model=umap_model
            )

            # Normalize coordinates for better visualization
            normalized_data = normalize_coordinates(reduced_data, scale=10.0)

            # Extract 3D coordinates
            coords_3d = [item['vector_3d'] for item in normalized_data]

            # 3. Run K-means clustering
            n_clusters = min(5, len(embedding_vectors))  # Max 5 clusters or fewer if not enough data
            print(f"Running K-means clustering with {n_clusters} clusters...")

            cluster_results = cluster_conversations(
                coords_3d,
                n_clusters=n_clusters,
                save_model=True
            )

            # Generate cluster names from conversation titles & topics
            from backend.services.clusterer import generate_cluster_names_from_topics
            cluster_names = generate_cluster_names_from_topics(
                cluster_results,
                all_topics,
                all_titles=all_titles
            )

            # 4. Update database with new coordinates and clusters
            print("Updating database with new coordinates and clusters...")
            updated_count = 0

            for i, conv_id in enumerate(conversation_ids):
                # Get updated embedding data
                embedding = db.query(Embedding).filter(
                    Embedding.conversation_id == conv_id
                ).first()

                if embedding:
                    # Update 3D coordinates
                    embedding.vector_3d = normalized_data[i]['vector_3d']
                    embedding.start_x = normalized_data[i]['start_x']
                    embedding.start_y = normalized_data[i]['start_y']
                    embedding.start_z = normalized_data[i]['start_z']
                    embedding.end_x = normalized_data[i]['end_x']
                    embedding.end_y = normalized_data[i]['end_y']
                    embedding.end_z = normalized_data[i]['end_z']
                    embedding.magnitude = normalized_data[i]['magnitude']

                # Update conversation cluster info
                conversation = db.query(Conversation).filter(
                    Conversation.id == conv_id
                ).first()

                if conversation:
                    cluster_id = cluster_results[i]['cluster_id']
                    conversation.cluster_id = cluster_id
                    conversation.cluster_name = cluster_names.get(
                        cluster_id,
                        f"Cluster {cluster_id}"
                    )
                    updated_count += 1

            db.commit()

            processing_time = (time.time() - start_time) * 1000

            # Get cluster statistics
            from backend.services.clusterer import get_cluster_statistics
            cluster_stats = get_cluster_statistics(cluster_results)

            return {
                "success": True,
                "conversations_processed": len(conversation_ids),
                "conversations_updated": updated_count,
                "n_clusters": n_clusters,
                "cluster_statistics": cluster_stats,
                "processing_time_ms": processing_time
            }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Reprocessing failed: {str(e)}"
        )
