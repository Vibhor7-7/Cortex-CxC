# CORTEX

**Your AI conversations, visualized as a navigable 3D memory space.**

Cortex transforms your ChatGPT and Claude conversation exports into an interactive 3D semantic map. Search across all your AI history, explore clusters of related topics, and inject past context into new conversations via MCP.

![Status](https://img.shields.io/badge/status-beta-yellow)
![Tests](https://img.shields.io/badge/tests-70%2F70-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)

---

## What It Does

**Upload** your ChatGPT or Claude HTML exports. Cortex parses every message, generates embeddings locally, and maps your conversations into 3D space — similar topics cluster together automatically.

**Explore** your entire AI history as an interactive point cloud. Orbit around it, fly through it, click any node to read the full conversation. Spring physics and flow fields keep the visualization alive.

**Search** with hybrid semantic + keyword retrieval. Ask a natural language question, and Cortex finds the most relevant past conversations instantly.

**Retrieve context across models** via MCP integration. Claude Desktop can call into your Cortex memory mid-conversation — pull in context from old chats without copy-pasting.

---

## Architecture
```
Browser (Three.js)  ──REST──▶  FastAPI Backend  ──▶  SQLite + NumPy Vector Store
                                     ▲
                                     │
                               MCP Server (stdio)
                                     ▲
                                     │
                              Claude Desktop
```

| Layer | Tech |
|-------|------|
| Frontend | Vanilla JS, Three.js, custom WebGL shaders |
| Backend | FastAPI, SQLAlchemy, BeautifulSoup4 |
| Database | SQLite + NumPy cosine similarity vector store |
| Embeddings | `nomic-embed-text` via Ollama (768D, local) |
| Summarization | `Qwen 2.5` via Ollama (local) |
| Dim. Reduction | UMAP (768D → 3D) |
| Clustering | K-Means with semantic naming |
| Context Retrieval | Anthropic MCP SDK |
| Retrieval QA | Backboard.io relevance guard |

**Zero API keys required.** All embedding and summarization runs locally through Ollama.

---

## Key Features

🧠 **Multi-format parsing** — Handles ChatGPT and Claude HTML exports out of the box

🗺️ **3D visualization** — UMAP-projected conversation clusters with orbit and fly-through controls

🔍 **Hybrid search** — Semantic + keyword retrieval across your full conversation history

🔌 **MCP tools** — `search_memory` and `fetch_chat` let Claude pull context from your past conversations

📤 **Drag-and-drop upload** — Batch ingest with per-file progress tracking

🛡️ **Relevance guard** — Backboard.io filters low-quality results before injecting context into models

⚡ **Fully local** — No cloud dependencies, no API costs, everything runs on your machine

---

## Design Decisions

| Decision | Why |
|----------|-----|
| Ollama over OpenAI API | Zero cost, works offline, no API keys |
| UMAP over t-SNE | Better global structure for 3D visualization |
| SQLite over Postgres | Single-file DB, no external dependencies |
| NumPy over ChromaDB | Zero-dep, ChromaDB broke on Python 3.14 |
| Vanilla JS over React | No build step, direct Three.js integration |

---

## Roadmap

- [x] Chat parsing (ChatGPT + Claude)
- [x] Local embeddings + summarization
- [x] UMAP reduction + K-Means clustering
- [x] 3D visualization with Three.js
- [x] Hybrid search system
- [x] MCP server integration
- [x] File upload UI
- [ ] Live backend data in frontend
- [ ] Docker deployment
- [ ] Multi-user support
- [ ] Browser extension for auto-export
- [ ] Cross-model context routing (Orbit)

---

## Built By

**Divyam & Vibhor** · CxC AI Hackathon Winners

📺 [YouTube](https://www.youtube.com/@divcodez) · 💻 [GitHub](https://github.com/Vibhor7-7/Cortex-CxC)
