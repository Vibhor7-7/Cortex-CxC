from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.chats import _to_conversation_out
from backend.core.openai_client import get_openai_client
from backend.db import get_db_session
from backend.models import Conversation, OpenAIFile
from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.services.openai_vector_store import (
	extract_conversation_id,
	get_or_create_vector_store_id,
	search_vector_store,
)


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
	query = payload.query.strip()
	if not query:
		raise HTTPException(status_code=422, detail="query must not be empty")

	client = get_openai_client()
	vector_store_id = get_or_create_vector_store_id(client)

	results = search_vector_store(
		client,
		vector_store_id=vector_store_id,
		query=query,
		limit=payload.limit,
	)

	conversation_ids: list[str] = []
	scores_by_id: dict[str, float] = {}
	raw_by_id: dict[str, dict] = {}

	for r in results:
		conv_id = extract_conversation_id(r)
		if not conv_id:
			continue
		if conv_id not in scores_by_id:
			conversation_ids.append(conv_id)
		score = float(r.get("score") or 0.0)
		scores_by_id[conv_id] = max(scores_by_id.get(conv_id, 0.0), score)
		raw_by_id[conv_id] = r

	with get_db_session() as db:
		conversations = db.query(Conversation).filter(Conversation.id.in_(conversation_ids)).all()
		by_id = {c.id: c for c in conversations}

		final_results: list[SearchResult] = []
		for conv_id in conversation_ids:
			conv = by_id.get(conv_id)
			if not conv:
				continue
			final_results.append(
				SearchResult(
					conversation=_to_conversation_out(conv),
					score=float(scores_by_id.get(conv_id, 0.0)),
					raw=raw_by_id.get(conv_id),
				)
			)

		# Fallback mapping via stored OpenAI file ids if attributes aren't present
		if not final_results and results:
			file_ids = [r.get("file_id") for r in results if isinstance(r.get("file_id"), str)]
			if file_ids:
				mappings = db.query(OpenAIFile).filter(OpenAIFile.file_id.in_(file_ids)).all()
				mapped_ids = [m.conversation_id for m in mappings]
				convs = db.query(Conversation).filter(Conversation.id.in_(mapped_ids)).all()
				by_id2 = {c.id: c for c in convs}
				for m in mappings:
					conv = by_id2.get(m.conversation_id)
					if conv:
						final_results.append(
							SearchResult(
								conversation=_to_conversation_out(conv),
								score=0.0,
								raw=None,
							)
						)

		return SearchResponse(query=query, results=final_results)
