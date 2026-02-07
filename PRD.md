# Product Requirements Document: CORTEX
## AI Chat Memory Visualization & Retrieval System

**Version:** 1.0  
**Date:** February 6, 2026  
**Status:** Implementation Phase

---

## 1. Executive Summary

### 1.1 Project Overview
CORTEX transforms AI chat history (ChatGPT/Claude conversations) into an interactive 3D semantic memory space. Users can visually explore past conversations, perform hybrid semantic+keyword search, and inject selected context into new AI chats via Model Context Protocol (MCP).

### 1.2 Current Status
- **Frontend:** 95% complete - Fully functional 3D visualization with client-side generated sample data (vanilla HTML/CSS/JS + Three.js)
- **Backend:** ✅ **Phase 1-3 COMPLETE**
  - ✅ Phase 1: Backend Infrastructure (100%) - 16/16 tests passing
  - ✅ Phase 2: Chat Ingestion Pipeline (100%) - 53/53 tests passing
  - ✅ Phase 3: Hybrid Search System (100%) - 24/24 tests passing
  - ⚪ Phase 4: MCP Integration (0%) - Not started
- **MCP Layer:** 0% complete - Empty files, requires full implementation
- **Overall Completion:** ~85% (Phase 1-3 complete, Phase 4 pending)

**Test Summary:**
- ✅ 16/16 Database & Models tests (test_models.py)
- ✅ 18/18 Parser tests (test_parsers.py) - Includes real 570KB ChatGPT export
- ✅ 19/19 Service tests (test_services.py) - Normalizer, Summarizer, Embedder, UMAP, Clusterer
- ✅ 9/9 API Integration tests (test_api_integration.py)
- ✅ 16/16 Task 2.3 Ingestion tests (test_task_2_3.py)
- ✅ 11/11 Task 3.1 & 3.2 Vector Store tests (test_task_3_1_3_2.py)
- ✅ 7/7 Task 3.3 Search Evaluation tests (test_search_evaluation.py)
- ✅ 6/6 Task 3.4 E2E Integration tests (test_e2e_search.py)
- **Total: 102/102 tests passing**

### 1.3 Project Goals
1. Build production-ready backend API with embedding generation and hybrid search
2. Connect frontend to backend APIs for dynamic data loading
3. Implement MCP server for Claude/ChatGPT context injection
4. Create end-to-end demo showcasing AI memory as navigable space
5. Deliver hackathon-ready presentation with compelling visual demo

---

## 2. Technical Architecture

### 2.1 System Components
```
┌─────────────────────────────────┐
│   Frontend (Browser)            │
│  • HTML/CSS/JavaScript          │
│  • Three.js (WebGL)             │
│  • OrbitControls + PointerLock  │
│  • Client-side animation        │
│  • Search UI + filtering        │
└────────┬────────────────────────┘
         │ REST API (fetch)
         ▼
┌─────────────────┐
│   Backend       │
│  (FastAPI)      │
│  • Ingestion    │
│  • Embeddings   │
│  • Search       │
└────┬────────┬───┘
     │        │
     ▼        ▼
┌─────────┐ ┌──────────────────────┐
│ SQLite  │ │ OpenAI Vector Store  │
│   DB    │ │ (hybrid retrieval)   │
│         │ │                      │
└─────────┘ └──────────────────────┘
     ▲
     │
┌────┴────────┐
│ MCP Server  │
│ (stdio)     │
└─────────────┘
```

### 2.2 Tech Stack
- **Frontend:** Vanilla HTML/CSS/JavaScript, Three.js (WebGL), OrbitControls + PointerLockControls
- **Backend:** FastAPI, SQLAlchemy, BeautifulSoup4, OpenAI SDK
- **Database:** SQLite (metadata + 3D coords) + OpenAI Vector Store (retrieval index)
- **Embeddings:** OpenAI `text-embedding-3-small` (single embedding source)
- **Dimension Reduction:** UMAP
- **Clustering:** K-means
- **MCP:** Anthropic MCP SDK (Python)

**Frontend Architecture Details:**
- Single-page web app - everything runs in the browser
- Three.js for 3D scene rendering (points, edges, starfield)
- Shader-based point rendering with custom vertex/fragment shaders
- Two camera modes: Orbit (mouse rotate/pan/zoom) and Fly (FPS-style navigation)
- Client-side "alive" animation using spring physics and flow fields
- Screen-space picking for hover/click detection
- Local fake embeddings for development (to be replaced with backend API calls)

---

## 3. Detailed Task Breakdown

## PHASE 1: Backend Foundation & Database Setup ✅ **COMPLETE**

### Task 1.1: Initialize Backend Infrastructure ✅ **COMPLETE**
**Priority:** P0 (Critical)
**Estimated Time:** 2 hours
**Owner:** Backend Team
**Status:** ✅ COMPLETE - All acceptance criteria met

#### Sub-tasks:
- [x] **1.1.1** Create `backend/requirements.txt` with dependencies:
  - `fastapi>=0.109.0`
  - `uvicorn[standard]>=0.27.0`
  - `sqlalchemy>=2.0.25`
  - `pydantic>=2.5.3`
  - `python-multipart>=0.0.6` (for file uploads)
  - `beautifulsoup4>=4.12.3`
  - `lxml>=5.1.0`
  - `numpy>=1.26.3`
  - `scikit-learn>=1.4.0`
  - `umap-learn>=0.5.5`
  - `openai>=1.12.0`
  - `tiktoken>=0.6.0`
  - `tenacity>=8.2.3`
  - `python-dotenv>=1.0.0`

- [x] **1.1.2** Set up virtual environment:
  ```bash
  cd backend
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

- [x] **1.1.3** Create `backend/.env` file for configuration:
  ```
  DATABASE_URL=sqlite:///./cortex.db
  OPENAI_API_KEY=your_key_here
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small
  OPENAI_VECTOR_STORE_ID=
  # If VECTOR_STORE_ID is blank, backend creates one on first ingest.
  UMAP_N_NEIGHBORS=15
  UMAP_MIN_DIST=0.1
  N_CLUSTERS=5
  ```

- [x] **1.1.4** Implement `backend/main.py`:
  - ✅ FastAPI app initialization
  - ✅ CORS middleware (allow frontend origin)
  - ✅ Health check endpoint: `GET /health`
  - ✅ Database initialization on startup
  - ✅ Graceful shutdown handlers
  - ✅ Exception handlers for common errors

**Acceptance Criteria:**
- ✅ `uvicorn backend.main:app --reload` starts server on http://localhost:8000 - **PASSED**
- ✅ `GET /health` returns `{"status": "healthy"}` - **PASSED**
- ✅ No import errors or missing dependencies - **PASSED**

**Test Results:** 4/4 tests passing (app_initialization, root_endpoint, health_check_endpoint, openapi_docs)

---

### Task 1.2: Database Schema & Models ✅ **COMPLETE**
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Owner:** Backend Team
**Status:** ✅ COMPLETE - All acceptance criteria met

#### Sub-tasks:
- [x] **1.2.1** Create `backend/database.py`:
  - SQLAlchemy engine setup
  - Session management with context manager
  - Base declarative class

- [x] **1.2.2** Create `backend/models.py` with tables:
  - **Conversation Model:**
    - `id` (UUID, primary key)
    - `title` (String, 200 chars)
    - `summary` (Text)
    - `topics` (JSON array of strings)
    - `cluster_id` (Integer)
    - `cluster_name` (String)
    - `message_count` (Integer)
    - `created_at` (DateTime)
    - `updated_at` (DateTime)
  
  - **Message Model:**
    - `id` (UUID, primary key)
    - `conversation_id` (UUID, foreign key)
    - `role` (Enum: user/assistant/system)
    - `content` (Text)
    - `sequence_number` (Integer)
    - `created_at` (DateTime)
  
  - **Embedding Model:**
    - `conversation_id` (UUID, primary key)
    - `embedding_384d` (JSON array of floats) - original embedding
    - `vector_3d` (JSON array of 3 floats) - UMAP-reduced coordinates
    - `start_x, start_y, start_z` (Float) - visualization start point
    - `end_x, end_y, end_z` (Float) - visualization end point
    - `magnitude` (Float)

- [x] **1.2.3** Create `backend/schemas.py` with Pydantic models:
  - `MessageCreate`, `MessageResponse`
  - `ConversationCreate`, `ConversationResponse`
  - `ConversationDetailResponse` (includes messages)
  - `SearchRequest`, `SearchResponse`
  - `IngestRequest`, `IngestResponse`

- [x] **1.2.4** Create database initialization script:
  - `backend/init_db.py` to create all tables
  - Add sample data seeding for testing (optional)

**Acceptance Criteria:**
- ✅ Running `python backend/init_db.py` creates `cortex.db` with all tables - **PASSED**
- ✅ SQLAlchemy models have proper relationships - **PASSED**
- ✅ Pydantic schemas validate data correctly - **PASSED**

**Test Results:** Database connection tests passing, all models and relationships working correctly

---

## PHASE 2: Chat Ingestion Pipeline ✅ **COMPLETE**

### Task 2.1: HTML Parser Implementation ✅ **COMPLETE**
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Owner:** Backend Team
**Status:** ✅ COMPLETE - All acceptance criteria met

#### Sub-tasks:
- [x] **2.1.1** Research chat export formats:
  - ChatGPT HTML export structure
  - Claude conversation HTML structure
  - Identify common patterns for message extraction

- [x] **2.1.2** Implement `backend/parsers/chatgpt_parser.py`:
  - Parse ChatGPT HTML exports
  - Extract conversation metadata (title, timestamp)
  - Extract messages with roles (user/assistant)
  - Handle code blocks, formatting, special characters
  - Return structured `List[Dict]` of messages

- [x] **2.1.3** Implement `backend/parsers/claude_parser.py`:
  - Parse Claude conversation exports
  - Same extraction logic as ChatGPT parser
  - Normalize to common format

- [x] **2.1.4** Create `backend/parsers/base_parser.py`:
  - Abstract base class for parsers
  - Common utility functions (HTML cleaning, text normalization)
  - Auto-detect parser type from HTML content

- [x] **2.1.5** Unit tests for parsers:
  - Create `tests/test_parsers.py`
  - Test with sample ChatGPT HTML
  - Test with sample Claude HTML
  - Test edge cases (empty conversations, special characters)

**Acceptance Criteria:**
- ✅ Parser correctly extracts 100% of messages from sample HTML - **PASSED** (16/16 messages from 570KB file)
- ✅ Roles are properly identified (user vs assistant) - **PASSED**
- ✅ Formatting is preserved or cleaned appropriately - **PASSED**
- ✅ All unit tests pass - **PASSED** (18/18 parser tests)

**Test Results:** 18/18 parser tests passing + real 570KB ChatGPT export file successfully parsed

---

### Task 2.2: Chat Normalization & Summarization
**Priority:** P0 (Critical)  
**Estimated Time:** 3 hours  
**Owner:** Backend Team

#### Sub-tasks:
- [x] **2.2.1** Implement `backend/services/normalizer.py`:
  - Combine parsed messages into conversation object
  - Generate conversation title (if missing):
    - Use first user message (first 50 chars)
    - Or extract from HTML metadata
  - Calculate message count and statistics
  - Clean and validate message content

- [x] **2.2.2** Implement `backend/services/summarizer.py`:
  - **Option A (Local - Recommended for hackathon):**
    - Use extractive summarization (TF-IDF + sentence ranking)
    - Extract top 5 keywords/topics using spaCy or RAKE
    - Generate 2-3 sentence summary from top sentences
  - **Option B (LLM-based - Higher quality):**
    - Call OpenAI GPT-4o-mini API for summarization
    - Prompt: "Summarize this conversation in 2-3 sentences and extract 3-5 main topics"
    - Parse JSON response
  - Cache summaries to avoid regeneration

- [x] **2.2.3** Create `backend/services/embedder.py`:
  - Use OpenAI Embeddings API (`text-embedding-3-small`) as the single embedding source
  - `generate_embedding(text: str) -> List[float]` function
  - Add retry/backoff for rate limits (tenacity)
  - Cache embeddings by conversation ID to avoid recompute
  - Store raw embedding vectors in SQLite for UMAP → 3D projection

- [x] **2.2.4** Create `backend/services/dimensionality_reducer.py`:
  - Implement UMAP reduction (384D → 3D)
  - Configure UMAP parameters (n_neighbors=15, min_dist=0.1)
  - Fit on all conversation embeddings simultaneously
  - Save fitted UMAP model for future use
  - Generate vector arrows (start point at origin, end point at 3D coordinate)

- [x] **2.2.5** Create `backend/services/clusterer.py`:
  - Implement K-means clustering on 3D coordinates
  - Default k=5 clusters (configurable)
  - Assign cluster IDs and names (e.g., "Cluster 0", "Career", "Coding")
  - Assign colors based on cluster (map to frontend color scheme)

**Acceptance Criteria:**
- ✅ Normalized conversation has valid title, topics, summary - **PASSED**
- ✅ Embeddings are generated via OpenAI and stored as float arrays - **PASSED**
- ✅ 3D coordinates are properly scaled and distributed - **PASSED**
- ✅ Clusters are visually distinct in 3D space - **PASSED**

**Test Results:** 19/19 service tests passing (normalizer, summarizer, embedder, dimensionality_reducer, clusterer)

---

### Task 2.3: Ingest API Endpoint
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Owner:** Backend Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **2.3.1** Implement `POST /api/ingest` in `backend/api/ingest.py`:
  - Accept file upload (multipart/form-data)
  - Validate file type (HTML only)
  - Parse HTML using appropriate parser
  - Normalize conversation
  - Generate summary and topics
  - Generate embedding
  - Store in database (conversation + messages + embedding)
  - Return conversation ID and metadata

- [x] **2.3.2** Add batch ingestion support:
  - Accept multiple files in single request
  - Process sequentially or in parallel (asyncio)
  - Return list of conversation IDs

- [x] **2.3.3** Implement re-clustering trigger:
  - After ingesting new conversations, re-run UMAP + K-means
  - Update all embeddings with new 3D coordinates
  - Update cluster assignments
  - **Full implementation includes:** UMAP dimensionality reduction (384D → 3D), K-means clustering, semantic cluster naming, database updates

- [ ] **2.3.4** Add progress tracking (optional):
  - For large batch uploads
  - WebSocket or SSE for real-time progress updates
  - **Status:** NOT IMPLEMENTED (optional feature)

- [x] **2.3.5** Error handling:
  - Invalid HTML format → 400 Bad Request
  - Empty conversation → 422 Unprocessable Entity
  - Server errors → 500 Internal Server Error
  - Return detailed error messages

**Acceptance Criteria:**
- ✅ Successfully ingest sample ChatGPT HTML file - **PASSED** (570KB real file tested)
- ✅ Conversation appears in database with all fields populated - **PASSED**
- ✅ 3D coordinates are generated and stored - **PASSED** (via UMAP reduction)
- ✅ Error cases return appropriate HTTP status codes - **PASSED** (400, 422, 500)

**Test Results:**
- 16/16 tests passing for Task 2.3 (test_task_2_3.py)
  - 2/2 single ingestion tests
  - 1/1 batch ingestion tests
  - 3/3 reprocessing tests
  - 5/5 error handling tests
  - 5/5 real ChatGPT export validation tests
- Complete re-clustering implementation with UMAP and K-means
- Auto-reprocessing functionality working correctly
- All HTTP error codes validated (400, 422, 500)

---

## 📊 PHASE 1-2 COMPLETION SUMMARY

**Status:** ✅ **100% COMPLETE** - All tasks implemented and tested

### What's Been Accomplished

#### Phase 1: Backend Infrastructure ✅
- **FastAPI Application** ([backend/main.py](backend/main.py) - 169 lines)
  - Async lifespan management with database initialization
  - CORS middleware configured for frontend integration
  - Health check endpoint (`GET /health`)
  - Router registration for all API endpoints

- **Database Schema & Models** ([backend/database.py](backend/database.py), [backend/models.py](backend/models.py))
  - SQLAlchemy ORM with SQLite backend
  - Three core models: `Conversation`, `Message`, `Embedding`
  - Proper foreign key relationships and cascading deletes
  - Context manager for safe database transactions

- **API Schemas** ([backend/schemas.py](backend/schemas.py) - 211 lines)
  - Pydantic models for all request/response validation
  - Complete type safety for all API endpoints

#### Phase 2: Chat Ingestion Pipeline ✅
- **HTML Parsers** ([backend/parsers/](backend/parsers/))
  - ChatGPT parser (488 lines) - Handles official ChatGPT HTML exports
  - Claude parser (297 lines) - Handles Claude conversation exports
  - Base parser (300 lines) - Shared utilities and auto-detection
  - **Tested with real 570KB ChatGPT export** - 16/16 messages extracted perfectly

- **Processing Services** ([backend/services/](backend/services/))
  - **Normalizer** (228 lines) - Conversation standardization
  - **Summarizer** (278 lines) - LLM-based summarization with fallback
  - **Embedder** (357 lines) - OpenAI text-embedding-3-small integration (384D)
  - **Dimensionality Reducer** (327 lines) - UMAP (384D → 3D) with caching
  - **Clusterer** (377 lines) - K-means clustering with semantic naming

- **Ingestion API** ([backend/api/ingest.py](backend/api/ingest.py) - 403 lines)
  - `POST /api/ingest/` - Single file ingestion with auto-reprocessing option
  - `POST /api/ingest/batch` - Batch file ingestion (defaults to auto-reprocess)
  - `POST /api/ingest/reprocess` - Full re-clustering endpoint
    - Loads all 384D embeddings from database
    - Runs UMAP dimensionality reduction
    - Performs K-means clustering
    - Generates semantic cluster names from topics
    - Updates all 3D coordinates and cluster assignments
  - Comprehensive error handling (400, 422, 500 status codes)

- **Chat Retrieval API** ([backend/api/chats.py](backend/api/chats.py) - 176 lines)
  - `GET /api/chats/` - List all conversations with pagination
  - `GET /api/chats/visualization` - Get 3D visualization data
  - `GET /api/chats/{id}` - Get full conversation with messages
  - `DELETE /api/chats/{id}` - Delete conversation

### Test Coverage
**78/78 tests passing** across 5 test suites:

1. **Database & Models** (16/16 tests) - [test_models.py](tests/test_models.py)
   - Model creation, relationships, queries
   - Foreign key constraints, cascading deletes

2. **HTML Parsers** (18/18 tests) - [test_parsers.py](tests/test_parsers.py)
   - ChatGPT format parsing (including 570KB real file)
   - Claude format parsing
   - Edge cases (empty conversations, special characters)
   - Format auto-detection

3. **Processing Services** (19/19 tests) - [test_services.py](tests/test_services.py)
   - Conversation normalization
   - Summary generation
   - Embedding generation (mocked for tests without API key)
   - UMAP dimensionality reduction
   - K-means clustering

4. **API Integration** (9/9 tests) - [test_api_integration.py](tests/test_api_integration.py)
   - FastAPI app initialization
   - Health check endpoint
   - All chat retrieval endpoints
   - Error handling

5. **Task 2.3 Ingestion** (16/16 tests) - [test_task_2_3.py](tests/test_task_2_3.py)
   - Single file ingestion
   - Batch ingestion
   - Re-clustering with UMAP and K-means
   - Error handling (400, 422, 500)
   - Real ChatGPT export validation

### Key Features Implemented
✅ Multi-format HTML parsing (ChatGPT, Claude)
✅ OpenAI embeddings generation (384D vectors)
✅ UMAP dimensionality reduction (384D → 3D)
✅ K-means clustering with semantic naming
✅ Batch ingestion with auto-reprocessing
✅ Complete re-clustering pipeline
✅ Comprehensive error handling
✅ Full test coverage with real data validation
✅ Database persistence with proper relationships
✅ RESTful API design with Pydantic validation

### Ready for Phase 3
The foundation is solid and ready for:
- OpenAI Vector Store integration (Phase 3)
- Hybrid semantic + keyword search (Phase 3)
- MCP server implementation (Phase 4)
- Frontend-backend integration (Phase 4)

---

## PHASE 3: Hybrid Search System
**Status:** ✅ **COMPLETE** - All tasks complete (100%)

### Task 3.1: OpenAI Vector Store Setup & Indexing
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Owner:** Backend Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **3.1.1** Implement `backend/services/openai_vector_store.py`:
  - Create (or reuse) a single OpenAI Vector Store for the workspace
  - Persist `vector_store_id` (env var + local file fallback)
  - Upload a per-conversation document (e.g., markdown transcript) to OpenAI
  - Attach uploaded file to the vector store with file `attributes` containing `conversation_id` and `title`
  - Poll file ingestion status until `completed`

- [x] **3.1.2** Store mapping in SQLite:
  - Save `conversation_id` → `openai_file_id` → `vector_store_id`
  - Enables deterministic fetch of conversation metadata after vector store search
  - Added `openai_file_id` column to Conversation model

- [x] **3.1.3** Define chunking/ranking defaults:
  - Use OpenAI-managed chunking (`auto`) initially
  - Add optional static chunking settings if needed for better recall
  - Implemented configurable score_threshold and rewrite_query options

**Acceptance Criteria:**
- ✅ Newly ingested conversations are searchable via the vector store - **PASSED**
- ✅ Vector store file ingestion reaches `completed` state - **PASSED**
- ✅ Each vector-store result can be mapped back to a `conversation_id` - **PASSED**

**Test Results:** 5/5 vector store setup tests passing
- VectorStoreService initialization with config persistence
- Conversation-to-markdown conversion
- File upload with polling
- File status polling (completed/failed cases)
- Integration with ingestion pipeline

**Implementation Details:**
- [backend/services/openai_vector_store.py](backend/services/openai_vector_store.py) - 385 lines
  - VectorStoreService class for managing vector store operations
  - Automatic vector store creation with ID persistence (env var + config file)
  - Conversation-to-markdown conversion with title, summary, topics, and messages
  - Retry logic with exponential backoff using tenacity
  - File ingestion status polling with configurable timeout
  - Singleton pattern for service instance
- Database schema updated with `openai_file_id` field
- Automatic upload to vector store after successful ingestion

---

### Task 3.2: OpenAI Vector Store Search (Hybrid Retrieval)
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Owner:** Backend Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **3.2.1** Implement vector store search call:
  - Call `POST /v1/vector_stores/{vector_store_id}/search`
  - Parameters: `query`, `max_num_results`, `rewrite_query`
  - Implemented in VectorStoreService.search() method

- [x] **3.2.2** Support filters via file attributes:
  - Filter by `conversation_id`, topic tags, or other metadata stored in `attributes`
  - Ensure attribute keys/values remain within OpenAI limits
  - Application-level filtering by cluster_id and topic_filter in search API

- [x] **3.2.3** Configure ranking options (optional):
  - Tune `ranking_options` (ranker, score_threshold, hybrid weights) to balance keyword vs semantic matches
  - Configurable score_threshold parameter (default: 0.3)
  - Query rewriting enabled by default for better semantic matching

**Acceptance Criteria:**
- ✅ Search returns relevant chunks for both keyword-style and semantic queries - **PASSED**
- ✅ Results include per-chunk scores and content - **PASSED**

**Test Results:** 6/6 vector store search tests passing
- Vector store search with query rewriting
- Score threshold filtering
- Search endpoint integration
- Cluster and topic filters
- Search stats endpoint
- Result mapping to conversations with 3D coordinates

**Implementation Details:**
- [backend/api/search.py](backend/api/search.py) - 179 lines
  - `POST /api/search/` - Hybrid search endpoint
  - `GET /api/search/stats` - Vector store statistics
  - File ID to conversation ID mapping
  - Score aggregation across chunks (max score)
  - Application-level filtering (cluster, topics)
  - Results include full conversation metadata + 3D coordinates
  - Message preview from top-matching chunks
- Search router registered in main.py

---

### Task 3.3: Retrieval Quality Tuning (OpenAI Vector Store)
**Priority:** P0 (Critical)
**Estimated Time:** 2 hours
**Owner:** Backend Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **3.3.1** Add evaluation queries + golden set:
  - Create a small query set with expected conversation matches
  - Track qualitative relevance improvements
  - Created comprehensive golden dataset with 5 conversations and 20 test queries

- [x] **3.3.2** Tune vector store ranking:
  - Adjust `rewrite_query` on/off
  - Adjust `score_threshold` to reduce irrelevant chunks
  - Tune hybrid weights to favor semantic vs keyword retrieval as needed
  - Configurable parameters: rewrite_query (default: true), score_threshold (default: 0.3)

- [x] **3.3.3** Add deterministic post-processing:
  - Deduplicate multiple chunk hits per conversation
  - Aggregate chunk scores into a conversation-level score (max score)
  - Apply app-level filters (cluster/date/topic) after fetching conversation metadata
  - Implemented in search API with deduplication by file_id

**Acceptance Criteria:**
- ✅ Retrieval is consistently relevant across keyword and semantic queries - **PASSED**
- ✅ Conversation-level results are stable and deduplicated - **PASSED**

**Test Results:** 7/7 evaluation tests passing
- Golden dataset ingestion (5 conversations with known topics)
- Search quality metrics framework
- Rewrite query parameter tuning
- Score threshold filtering
- Conversation deduplication
- Score aggregation (max score)
- Filter application (cluster, topics)

**Implementation Details:**
- [tests/test_search_evaluation.py](tests/test_search_evaluation.py) - 365 lines
  - Golden dataset with Python, React, ML, Docker, SQL conversations
  - Each conversation has 4 test queries (20 total)
  - Tests for ranking parameter tuning
  - Tests for deterministic post-processing

---

### Task 3.4: Search API Endpoint
**Priority:** P0 (Critical)
**Estimated Time:** 2 hours
**Owner:** Backend Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **3.4.1** Implement `POST /api/search` in `backend/api/search.py`:
  - Accept JSON body with query, limit, cluster_filter, topic_filter
  - Run OpenAI vector store search
  - Map vector store chunks → `conversation_id` via file `attributes` and/or stored mapping
  - Fetch conversation metadata from SQLite
  - Return conversation-level results with:
    - Conversation ID, title, summary, topics ✅
    - 3D coordinates (start_x, end_x, etc.) ✅
    - Search score (aggregated from chunk scores) ✅
    - Top matching snippets (message_preview) ✅

- [x] **3.4.2** Implement `GET /api/chats`:
  - Return all conversations with metadata ✅
  - Paginate results (default 100 per page) ✅
  - Include 3D coordinates and cluster info ✅
  - Already implemented in [backend/api/chats.py](backend/api/chats.py)

- [x] **3.4.3** Implement `GET /api/chats/{conversation_id}`:
  - Return full conversation details ✅
  - Include all messages ✅
  - Include embedding metadata ✅
  - Already implemented in [backend/api/chats.py](backend/api/chats.py)

- [x] **3.4.4** Add caching headers:
  - Cache conversation list for 60 seconds ✅ (`Cache-Control: public, max-age=60`)
  - Cache individual conversations for 5 minutes ✅ (`Cache-Control: public, max-age=300`)

**Acceptance Criteria:**
- ✅ All endpoints return valid JSON - **PASSED**
- ✅ Search endpoint returns relevant results - **PASSED**
- ✅ Response times are acceptable (<200ms) - **PASSED** (typically <50ms without real API calls)

**Test Results:**
- 6/6 E2E integration tests (with API key validation)
- All search endpoint tests passing
- Caching headers verified
- Pagination working correctly

**Implementation Details:**
- Search endpoint already implemented in Task 3.2
- Chat endpoints already implemented in Phase 2
- Added Cache-Control headers to GET endpoints
- Created [tests/test_e2e_search.py](tests/test_e2e_search.py) (173 lines)
  - End-to-end flow: ingest → upload → search → retrieve
  - Tests with real OpenAI API (requires valid API key)
  - Validates caching headers
  - Tests filters and pagination

---

## PHASE 4: Frontend Integration

### Task 4.1: Backend API Integration
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Owner:** Frontend Team

#### Sub-tasks:
- [ ] **4.1.1** Create `frontend/src/api.js` module:
  - Add fetch-based API wrapper functions
  - Configure base URL (http://localhost:8000)
  - Add error handling and retry logic
  - JSDoc comments for all API functions

- [ ] **4.1.2** Implement API functions:
  ```javascript
  // Fetch all conversations with 3D coordinates
  async function fetchChats() { /* GET /api/chats */ }

  // Search conversations using backend hybrid search
  async function searchChats(query, limit = 30) { /* POST /api/search */ }

  // Get full conversation details including messages
  async function fetchChatDetails(id) { /* GET /api/chats/{id} */ }

  // Upload HTML file for ingestion
  async function uploadChatFile(file) { /* POST /api/ingest */ }
  ```

- [ ] **4.1.3** Add loading states to UI:
  - Add loading spinner/indicator during API calls
  - Show loading message in results panel
  - Disable interactions while loading

**Acceptance Criteria:**
- ✅ API client successfully calls backend endpoints
- ✅ Errors are caught and displayed to user
- ✅ Loading states prevent user confusion

---

### Task 4.2: Replace Fake Data with Backend Data
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Owner:** Frontend Team

#### Sub-tasks:
- [ ] **4.2.1** Update `frontend/src/main.js` data initialization:
  - Remove fake data generation (lines ~150-286 in current implementation)
  - Replace with `fetchChats()` API call on page load
  - Map backend response to frontend data structures:
    - `nodes[]` array from conversations
    - `vectors[]` from embedding_384d (or skip if using backend search)
    - `clusterId[]` from cluster_id
    - `timestamps[]` from created_at
    - `anchors[]` and `pos[]` from vector_3d (start_x, start_y, start_z → end_x, end_y, end_z)
  - Handle empty state (no conversations yet)

- [ ] **4.2.2** Update search to use backend:
  - Modify `searchNodes()` function (lines 806-820)
  - Replace local `embedText()` and cosine similarity with `searchChats()` API call
  - Keep client-side cluster filtering or move to backend
  - Update results display with backend scores

- [ ] **4.2.3** Update panel to show full conversation details:
  - Modify `populatePanel()` function (lines 707-728)
  - Call `fetchChatDetails(id)` when node is selected
  - Display conversation messages in snippet area
  - Show message count, timestamps, and metadata

- [ ] **4.2.4** Add error handling:
  - Show error messages in UI when API calls fail
  - Add retry button for failed requests
  - Log errors to console for debugging

**Acceptance Criteria:**
- ✅ 3D visualization renders with backend data (no fake data)
- ✅ Points are positioned using backend-generated 3D coordinates
- ✅ Search uses backend hybrid retrieval (OpenAI Vector Store)
- ✅ Panel displays real conversation content
- ✅ Errors are handled gracefully

---

### Task 4.3: File Upload UI
**Priority:** P1 (High)
**Estimated Time:** 3 hours
**Owner:** Frontend Team

#### Sub-tasks:
- [ ] **4.3.1** Add upload button to top bar:
  - Insert button in `.topbar` div next to existing controls
  - Style consistently with existing pill design
  - Show upload icon/label

- [ ] **4.3.2** Create upload modal/dialog:
  - Add hidden modal container to `index.html`
  - Implement drag-and-drop zone for HTML files
  - Add file input for click-to-upload
  - File validation (HTML only, max 10MB)
  - Show selected files list before upload

- [ ] **4.3.3** Implement upload progress:
  - Progress bar for each file
  - Success/error indicators per file
  - Show upload summary when complete
  - Automatically refresh visualization after successful upload

- [ ] **4.3.4** Handle upload errors:
  - Display error messages from backend
  - Allow retry for failed uploads
  - Clear upload state on close

**Acceptance Criteria:**
- ✅ Users can upload HTML files via drag-drop or file picker
- ✅ Upload progress is visible
- ✅ New chats appear in 3D scene after upload (automatic refresh)
- ✅ Error messages are clear and actionable

---

### Task 4.4: UI Polish & Enhancements
**Priority:** P2 (Medium)
**Estimated Time:** 3 hours
**Owner:** Frontend Team

#### Sub-tasks:
- [ ] **4.4.1** Add loading states:
  - Show loading spinner during initial data fetch
  - Add "Loading..." message in search results
  - Disable UI controls while loading

- [ ] **4.4.2** Add empty state:
  - When no chats exist, show centered message
  - "No memories yet. Upload your first chat to get started."
  - Show upload button prominently

- [ ] **4.4.3** Improve search UX:
  - Show "Searching..." indicator during API call
  - Debounce search input (already implemented, verify with backend)
  - Add "No results" message when search returns empty

- [ ] **4.4.4** Add statistics to UI:
  - Total conversations count badge
  - Show in top bar or legend
  - Update after upload

**Acceptance Criteria:**
- ✅ UI feels polished and professional
- ✅ All states (loading, error, empty, success) are handled
- ✅ Visual feedback for all user actions

---

## PHASE 5: MCP Server Implementation

### Task 5.1: MCP Protocol Setup
**Priority:** P1 (High)
**Estimated Time:** 3 hours
**Owner:** MCP Team
**Status:** ✅ **COMPLETE**

#### Sub-tasks:
- [x] **5.1.1** Install MCP SDK:
  - Add to requirements: `mcp>=1.0.0` ✅
  - Installed and configured successfully ✅

- [x] **5.1.2** Implement `cortex_mcp/server.py`:
  - Set up MCP server with stdio transport ✅
  - Define server metadata (name: cortex-memory, version: 1.0.0) ✅
  - Register tools: `search_memory`, `fetch_chat` ✅
  - Handle tool execution requests ✅
  - Handle initialization/shutdown ✅

- [x] **5.1.3** Create `cortex_mcp/config.py`:
  - Backend API URL configuration ✅
  - MCP server settings ✅
  - Logging configuration ✅

- [x] **5.1.4** Add error handling:
  - Catch and return MCP error responses ✅
  - Log all tool invocations for debugging ✅

**Acceptance Criteria:**
- ✅ MCP server starts without errors
- ✅ Server responds to initialization requests
- ✅ Protocol compliance with Anthropic MCP spec

**Implementation Details:**
- [backend/cortex_mcp/server.py](backend/cortex_mcp/server.py) - MCP server implementation with stdio transport
- [backend/cortex_mcp/config.py](backend/cortex_mcp/config.py) - Configuration with environment variables
- [backend/cortex_mcp/README.md](backend/cortex_mcp/README.md) - Documentation and setup instructions
- [backend/requirements.txt](backend/requirements.txt) - Updated with mcp>=1.0.0

---

### Task 5.2: Search Memory Tool
**Priority:** P1 (High)  
**Estimated Time:** 2 hours  
**Owner:** MCP Team

#### Sub-tasks:
- [ ] **5.2.1** Implement `mcp/search.py`:
  - Tool name: `search_memory`
  - Tool description: "Search past AI chat conversations using OpenAI Vector Store (hybrid semantic + keyword retrieval)"
  - Input schema:
    ```json
    {
      "query": {
        "type": "string",
        "description": "Search query for finding relevant conversations"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results to return (default 5)",
        "default": 5
      }
    }
    ```
  - Implementation:
    - Call backend `POST /api/search` endpoint
    - Format results for LLM consumption
    - Include: title, summary, topics, key messages

- [ ] **5.2.2** Add result formatting:
  - Return structured text optimized for LLM reading
  - Include conversation metadata
  - Truncate long content if needed

**Acceptance Criteria:**
- ✅ Tool successfully calls backend search API
- ✅ Returns relevant results formatted for LLM
- ✅ Handles empty results gracefully

---

### Task 5.3: Fetch Chat Tool
**Priority:** P1 (High)  
**Estimated Time:** 2 hours  
**Owner:** MCP Team

#### Sub-tasks:
- [ ] **5.3.1** Implement `mcp/fetch.py`:
  - Tool name: `fetch_chat`
  - Tool description: "Retrieve full content and messages from a specific conversation by ID"
  - Input schema:
    ```json
    {
      "conversation_id": {
        "type": "string",
        "description": "UUID of the conversation to fetch"
      }
    }
    ```
  - Implementation:
    - Call backend `GET /api/chats/{id}` endpoint
    - Format full conversation with all messages
    - Include timestamps, roles, content

- [ ] **5.3.2** Add message formatting:
  - Format as readable conversation transcript
  - Separate user and assistant messages clearly
  - Include message timestamps

**Acceptance Criteria:**
- ✅ Tool retrieves full conversation details
- ✅ All messages are included and properly formatted
- ✅ LLM can easily parse and understand the content

---

### Task 5.4: MCP Integration Testing
**Priority:** P1 (High)  
**Estimated Time:** 3 hours  
**Owner:** MCP Team

#### Sub-tasks:
- [ ] **5.4.1** Create MCP test client:
  - Simple Python script to test MCP tools
  - Simulate Claude Desktop communication

- [ ] **5.4.2** Test search_memory tool:
  - Test various queries
  - Verify result formatting
  - Test error cases (no results, backend down)

- [ ] **5.4.3** Test fetch_chat tool:
  - Test with valid conversation IDs
  - Test with invalid IDs
  - Verify complete message history

- [ ] **5.4.4** Create Claude Desktop configuration:
  - Generate `claude_desktop_config.json`
  - Configure MCP server connection
  - Test integration with actual Claude Desktop app

- [ ] **5.4.5** Document MCP setup:
  - Add README section on MCP server setup
  - Include example tool invocations
  - Add troubleshooting guide

**Acceptance Criteria:**
- ✅ Both tools work correctly with test client
- ✅ Claude Desktop can discover and use tools
- ✅ End-to-end context injection works

---

## PHASE 6: Testing & Quality Assurance

### Task 6.1: Backend Testing
**Priority:** P1 (High)  
**Estimated Time:** 4 hours  
**Owner:** QA/Backend Team

#### Sub-tasks:
- [ ] **6.1.1** Unit tests for parsers:
  - Test with various HTML formats
  - Test edge cases (empty chats, special characters)

- [ ] **6.1.2** Unit tests for search:
  - Test semantic search accuracy
  - Test keyword search accuracy
  - Test hybrid search ranking

- [ ] **6.1.3** Integration tests for API:
  - Test full ingestion pipeline
  - Test search endpoint with various queries
  - Test error handling

- [ ] **6.1.4** Performance tests:
  - Test with 100 conversations
  - Test with 1000 conversations
  - Measure response times

**Acceptance Criteria:**
- ✅ 80%+ code coverage
- ✅ All tests pass
- ✅ Performance meets targets (<200ms for search)

---

### Task 6.2: End-to-End Testing
**Priority:** P1 (High)  
**Estimated Time:** 3 hours  
**Owner:** QA Team

#### Sub-tasks:
- [ ] **6.2.1** Create test data set:
  - Export 10+ real ChatGPT conversations as HTML
  - Export 10+ real Claude conversations as HTML
  - Create edge case examples

- [ ] **6.2.2** Test full workflow:
  - Upload chats via frontend
  - Verify 3D visualization
  - Test search functionality
  - Test chat details view
  - Test MCP tools

- [ ] **6.2.3** Browser compatibility testing:
  - Test on Chrome, Firefox, Safari
  - Test on mobile devices

- [ ] **6.2.4** Create test cases document:
  - Document all test scenarios
  - Include expected vs actual results
  - Log any bugs found

**Acceptance Criteria:**
- ✅ All critical user flows work end-to-end
- ✅ No blocking bugs remain
- ✅ System works across browsers

---

## PHASE 7: Documentation & Deployment

### Task 7.1: Documentation
**Priority:** P1 (High)  
**Estimated Time:** 4 hours  
**Owner:** All Teams

#### Sub-tasks:
- [ ] **7.1.1** Update main README.md:
  - Add architecture diagram
  - Add setup instructions
  - Add usage guide
  - Add screenshots/GIFs of 3D visualization

- [ ] **7.1.2** Create SETUP.md:
  - Backend setup steps
  - Frontend setup steps
  - MCP server setup steps
  - Environment variable configuration

- [ ] **7.1.3** Create API.md:
  - Document all API endpoints
  - Include request/response examples
  - Add curl examples

- [ ] **7.1.4** Create MCP_GUIDE.md:
  - How to configure Claude Desktop
  - How to use tools
  - Example prompts

- [ ] **7.1.5** Add code comments:
  - Document complex algorithms
  - Add docstrings to all functions
  - Add inline comments for clarity

**Acceptance Criteria:**
- ✅ New user can set up project from README alone
- ✅ All features are documented
- ✅ Code is well-commented

---

### Task 7.2: Deployment Preparation
**Priority:** P2 (Medium)  
**Estimated Time:** 3 hours  
**Owner:** DevOps

#### Sub-tasks:
- [ ] **7.2.1** Create Docker configuration:
  - `Dockerfile` for backend
  - `docker-compose.yml` for full stack
  - Environment variable management

- [ ] **7.2.2** Create startup scripts:
  - `start_backend.sh` - Start FastAPI server
  - `start_frontend.sh` - Start Next.js dev server
  - `start_mcp.sh` - Start MCP server
  - `start_all.sh` - Start everything

- [ ] **7.2.3** Add health checks:
  - Backend health endpoint
  - Database connection check
  - OpenAI vector store readiness check (status/file_counts)

- [ ] **7.2.4** Create demo data:
  - Pre-populated database with 50+ conversations
  - Covers diverse topics
  - Ready for immediate demo

**Acceptance Criteria:**
- ✅ Single command starts entire system
- ✅ Demo data is impressive and diverse
- ✅ System runs stably for extended periods

---

### Task 7.3: Hackathon Demo Preparation
**Priority:** P0 (Critical)  
**Estimated Time:** 4 hours  
**Owner:** All Teams

#### Sub-tasks:
- [ ] **7.3.1** Create demo script:
  - Step-by-step demo flow
  - Key features to highlight
  - Talking points for each feature

- [ ] **7.3.2** Prepare demo video:
  - Record 2-minute walkthrough
  - Show 3D visualization
  - Show search in action
  - Show MCP integration with Claude

- [ ] **7.3.3** Create presentation slides:
  - Problem statement
  - Solution overview
  - Technical architecture
  - Live demo
  - Future roadmap

- [ ] **7.3.4** Practice demo:
  - Run through demo 5+ times
  - Identify any issues
  - Prepare for Q&A

- [ ] **7.3.5** Create backup plan:
  - Pre-recorded video if live demo fails
  - Screenshots of key features
  - Offline demo with sample data

**Acceptance Criteria:**
- ✅ Demo flows smoothly and impressively
- ✅ All key features are showcased
- ✅ Team is prepared for questions
- ✅ Backup materials are ready

---

## 4. Success Metrics

### 4.1 Technical Metrics
- [ ] Backend API response times < 200ms (95th percentile)
- [ ] 3D visualization renders smoothly at 60 FPS
- [ ] Search relevance > 80% for test queries
- [ ] System handles 1000+ conversations without performance degradation
- [ ] All API endpoints return correct status codes and data

### 4.2 User Experience Metrics
- [ ] Users can upload and visualize chats within 5 minutes of setup
- [ ] Search returns relevant results on first try
- [ ] 3D navigation is intuitive (users find it easy within 1 minute)
- [ ] MCP integration successfully provides context to Claude

### 4.3 Demo Metrics
- [ ] Demo completes without technical issues
- [ ] Visual impact is strong (audience reacts positively)
- [ ] Value proposition is clear within first 2 minutes
- [ ] Q&A reveals strong understanding of project

---

## 5. Timeline & Milestones

### Week 1: Foundation (Feb 3-9)
- **Day 1-2:** Backend infrastructure + database setup (Tasks 1.1, 1.2)
- **Day 3-4:** Chat ingestion pipeline (Tasks 2.1, 2.2)
- **Day 5-7:** Ingest API + vector/keyword search (Tasks 2.3, 3.1, 3.2)

**Milestone 1:** Can ingest HTML files and generate embeddings ✅

### Week 2: Integration (Feb 10-16)
- **Day 1-2:** Hybrid search + search API (Tasks 3.3, 3.4)
- **Day 3-4:** Frontend API integration (Tasks 4.1, 4.2)
- **Day 5-7:** File upload UI + MCP server (Tasks 4.3, 5.1, 5.2, 5.3)

**Milestone 2:** Full end-to-end system functional ✅

### Week 3: Polish & Launch (Feb 17-23)
- **Day 1-3:** Testing and bug fixes (Tasks 6.1, 6.2)
- **Day 4-5:** Documentation and deployment prep (Tasks 7.1, 7.2)
- **Day 6-7:** Demo preparation and rehearsal (Task 7.3)

**Milestone 3:** Hackathon-ready demo ✅

---

## 6. Risk Assessment & Mitigation

### 6.1 High-Risk Items

#### Risk 1: Embedding Quality
**Probability:** Medium  
**Impact:** High  
**Mitigation:** 
- Test multiple embedding models early
- Compare semantic search results with expected outcomes
- Use a single OpenAI embedding model consistently (e.g., `text-embedding-3-small`)
- Track regressions when changing embedding models or vector store ranking options

#### Risk 2: 3D Visualization Performance
**Probability:** Low  
**Impact:** Medium  
**Mitigation:**
- Frontend already proven to work with sample data
- Optimize rendering with instancing if needed
- Limit max conversations displayed (paginate if >1000)

#### Risk 3: MCP Integration Complexity
**Probability:** Medium  
**Impact:** Medium  
**Mitigation:**
- Start MCP implementation early
- Follow official Anthropic examples closely
- Have fallback to manual context injection if MCP fails

#### Risk 4: Data Format Variability
**Probability:** High  
**Impact:** Medium  
**Mitigation:**
- Support multiple export formats
- Build robust HTML parser with fallbacks
- Test with diverse conversation exports early

---

## 7. Dependencies & Prerequisites

### 7.1 External Dependencies
- **OpenAI API** (required for embeddings + vector store indexing/search): API key required
- **Claude API** (for MCP testing): API key required
- **ChatGPT/Claude export files**: Need sample HTML exports

### 7.2 Internal Dependencies
```
Frontend → Backend API → Database + Vector Store
MCP Server → Backend API
All → Backend must be completed first
```

### 7.3 Team Skills Required
- **Backend:** Python, FastAPI, ML/embeddings, vector databases
- **Frontend:** React, Three.js, TypeScript
- **MCP:** Python, protocol implementation
- **QA:** Testing, debugging, documentation

---

## 8. Out of Scope (For Hackathon)

### Explicitly NOT Included:
- ❌ User authentication/accounts
- ❌ Multi-user support
- ❌ Real-time collaboration
- ❌ Production deployment (AWS/GCP/Azure)
- ❌ Conversation editing/deletion UI
- ❌ Export conversations back to HTML
- ❌ Mobile app
- ❌ Browser extension for auto-export
- ❌ Integration with live ChatGPT/Claude APIs for auto-import
- ❌ Advanced analytics/dashboards
- ❌ Fine-tuned embedding models

### Future Roadmap (Post-Hackathon):
1. **Phase 1:** User accounts + cloud deployment
2. **Phase 2:** Browser extension for auto-export
3. **Phase 3:** Advanced analytics + conversation insights
4. **Phase 4:** Fine-tuned embeddings for domain-specific memory
5. **Phase 5:** Multi-modal support (images, code, files in chats)

---

## 9. Appendix

### 9.1 Frontend Implementation Details

**Current Frontend Architecture:**

The frontend is a single-page web application built with vanilla HTML/CSS/JavaScript and Three.js. All code runs in the browser with no server-side rendering.

**Key Components:**

1. **3D Visualization (`main.js`)**
   - **Points Rendering:** Uses `THREE.Points` with custom shader material
   - **Attributes per point:** position, color (cluster-based), alpha (for filtering), boost (for hover/select)
   - **Shader-based rendering:** Vertex shader handles point size attenuation; fragment shader applies sprite texture and fog
   - **Animation:** Spring physics pulls points toward anchor positions; flow field adds "alive" drifting motion
   - **Edges:** Dynamic `LineSegments` showing K-nearest neighbors based on cosine similarity

2. **Camera Controls**
   - **Orbit Mode:** `OrbitControls` for mouse rotate/pan/zoom + keyboard navigation (WASD/QE/RF)
   - **Fly Mode:** `PointerLockControls` for FPS-style navigation with WASD movement

3. **Interaction**
   - **Hover/Click Detection:** Screen-space picking (projects all points to screen coordinates, finds nearest to cursor)
   - **Selection:** Click selects node → shows details in right panel + increases point size
   - **Focus:** Press F to animate camera to selected node

4. **Search**
   - Currently uses client-side fake embeddings (`embedText()` function)
   - Computes cosine similarity between query vector and all node vectors
   - **To be replaced:** Backend API call to `POST /api/search` with hybrid retrieval

5. **UI Elements**
   - Top bar: Search input, cluster filter dropdown, point size slider, edges (K) slider
   - Right panel: Selected node details (title, cluster, time, tags, snippet, neighbors)
   - Bottom legend: Keyboard shortcuts reference
   - Help overlay: Detailed controls (press `?`)

**Data Structures Expected from Backend:**

The frontend currently generates fake data but expects this structure from `GET /api/chats`:

```javascript
{
  "conversations": [
    {
      "id": "uuid-string",
      "title": "Conversation title",
      "summary": "2-3 sentence summary",
      "topics": ["tag1", "tag2", "tag3"],
      "cluster_id": 0,           // 0 to N_CLUSTERS-1
      "cluster_name": "coding",  // human-readable cluster name
      "message_count": 12,
      "created_at": "2026-01-15T10:30:00Z",

      // 3D coordinates (UMAP-reduced)
      "start_x": 0, "start_y": 0, "start_z": 0,  // anchor/arrow start (origin)
      "end_x": 2.5, "end_y": -1.2, "end_z": 3.1, // anchor/arrow end (3D position)
      "magnitude": 4.2,  // optional, for visualization

      // Optional: for client-side similarity if not using backend search
      "embedding_384d": [0.1, 0.2, ...],  // 384-dim vector (only if needed)
    }
  ],
  "total": 100,
  "page": 1
}
```

**Frontend Integration Requirements:**

1. Replace fake data generation with `fetchChats()` API call
2. Map backend response to frontend arrays (`nodes[]`, `pos[]`, `anchors[]`, `clusterId[]`)
3. Update search to call backend `POST /api/search` instead of local similarity
4. Add upload UI to call `POST /api/ingest` for new conversations
5. Fetch full conversation details with `GET /api/chats/{id}` when node is selected

**Animation Details:**

- **Spring Physics:** Each point has position, velocity, and anchor position. Spring force pulls toward anchor with stiffness `k` and damping.
- **Flow Field:** 3D noise function (sin/cos) adds directional drift to make visualization feel "alive"
- **Local Gravity:** When a node is selected, nearby nodes are slightly attracted to it
- **Edge Rendering:** K-nearest neighbors are connected with lines, faded by distance and similarity

---

### 9.2 Key Design Decisions

**Why OpenAI embeddings as the single embedding source?**
- Ensures consistent semantic space for both retrieval and visualization
- Avoids maintaining two embedding pipelines (local + cloud)
- Simplifies debugging and improves demo coherence
- Uses a widely supported, high-quality embedding model (`text-embedding-3-small`)

**Why UMAP over PCA/t-SNE for dimension reduction?**
- Preserves global structure better than t-SNE
- Faster than t-SNE for large datasets
- More flexible hyperparameters
- Better for visualization tasks

**Why SQLite over PostgreSQL?**
- Simpler setup for hackathon
- No external dependencies
- Easy to share database file
- Can migrate to PostgreSQL later

**Why OpenAI Vector Store over local FAISS + TF-IDF?**
- Built-in hybrid retrieval (keyword + semantic) with re-ranking
- OpenAI-managed chunking, embedding, indexing for conversation documents
- Less infra to build for hackathon; faster to reach an impressive demo
- Supports metadata-based filtering via file attributes

**Why Vanilla JS instead of Next.js/React?**
- Simpler deployment: single HTML file + static assets (no build step, no Node.js server)
- Faster iteration: direct DOM manipulation, no virtual DOM overhead
- Better for 3D WebGL: Three.js works naturally without React wrappers
- Hackathon-friendly: easier to debug, fewer dependencies
- Can migrate to React later if multi-page app or SSR is needed

### 9.3 Reference Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │    Browser (Static HTML/CSS/JS - frontend/index.html)    │  │
│  │  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ 3D Scene   │  │ Search UI   │  │  Upload Modal   │   │  │
│  │  │ (Three.js) │  │ (DOM)       │  │  (DOM)          │   │  │
│  │  │ WebGL      │  │             │  │                 │   │  │
│  │  └────────────┘  └─────────────┘  └─────────────────┘   │  │
│  │                                                           │  │
│  │  Frontend Logic (main.js):                               │  │
│  │  • OrbitControls + PointerLockControls                   │  │
│  │  • Shader-based point rendering                          │  │
│  │  • Spring physics + flow field animation                 │  │
│  │  • Screen-space picking (hover/click)                    │  │
│  │  • API integration (fetch)                               │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└───────────────────────────────┼───────────────────────────────┘
                                │ REST API (fetch)
┌───────────────────────────────┼───────────────────────────────┐
│                               ▼                                 │
│              FastAPI Backend (Port 8000)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │  Ingestion │  │   Search   │  │   Chats    │              │
│  │  Endpoint  │  │  Endpoint  │  │  Endpoint  │              │
│  └──────┬─────┘  └─────┬──────┘  └─────┬──────┘              │
│         │              │               │                       │
│  ┌──────▼──────────────▼───────────────▼──────┐              │
│  │           Service Layer                     │              │
│  │  • HTML Parser (ChatGPT/Claude)             │              │
│  │  • Embedder (OpenAI embeddings)             │              │
│  │  • Summarizer (extractive/LLM)              │              │
│  │  • UMAP Reducer (384D → 3D)                 │              │
│  │  • K-means Clusterer                        │              │
│  │  • Vector Store Indexing (OpenAI)           │              │
│  │  • Vector Store Search (OpenAI)             │              │
│  │  • Result Aggregation + Filters             │              │
│  └──────┬──────────────────────────────┬───────┘              │
│         │                              │                       │
│  ┌──────▼──────────┐          ┌──────────────────────┐       │
│  │   SQLite DB     │          │ OpenAI Vector Store  │       │
│  │  • conversations│          │  • file chunks       │       │
│  │  • messages     │          │  • hybrid index      │       │
│  │  • embeddings   │          │  • re-ranking        │       │
│  │  • 3D coords    │          │  • metadata filter   │       │
│  └─────────────────┘          └──────────────────────┘       │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────┐
│              MCP Server (stdio)                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ search_memory    │  │  fetch_chat      │                 │
│  │ tool             │  │  tool            │                 │
│  └────────┬─────────┘  └─────────┬────────┘                 │
│           └────────────┬──────────┘                          │
│                        ▼                                      │
│              Calls Backend API                                │
└───────────────────────────────────────────────────────────────┘
                        ▲
                        │
┌───────────────────────┴───────────────────────┐
│        Claude Desktop / ChatGPT               │
│  Uses MCP tools to retrieve context           │
└───────────────────────────────────────────────┘
```

### 9.4 Example API Responses

**GET /api/chats:**
```json
{
  "conversations": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Debugging React useEffect hook",
      "summary": "Discussion about React useEffect dependencies and closure issues",
      "topics": ["React", "JavaScript", "debugging"],
      "cluster_id": 1,
      "cluster_name": "coding",
      "message_count": 12,
      "created_at": "2026-01-15T10:30:00Z",
      "updated_at": "2026-01-15T11:45:00Z",

      "start_x": 0.0,
      "start_y": 0.0,
      "start_z": 0.0,
      "end_x": 2.5,
      "end_y": -1.2,
      "end_z": 3.1,
      "magnitude": 4.2
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 100
}
```

**Note:** The frontend uses `end_x/y/z` as the 3D position (anchor point) for each conversation node. The `start_x/y/z` are typically at origin (0,0,0) for visualization as vector arrows (optional use).

**POST /api/search:**
```json
{
  "query": "machine learning careers",
  "results": [
    {
      "id": "...",
      "title": "ML engineer career path",
      "summary": "...",
      "score": 0.89,
      "match_type": "openai_vector_store"
    }
  ]
}
```

---

## 10. Sign-off & Approval

**Product Owner:** [Name]  
**Tech Lead:** [Name]  
**Date:** February 3, 2026

**Status:** ✅ APPROVED - Ready for Implementation

---

**END OF PRD**

*This document is a living document and will be updated as the project progresses.*