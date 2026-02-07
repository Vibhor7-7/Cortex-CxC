# Cortex Backend Setup

## Environment Configuration

This project runs fully locally using Ollama + a numpy-based vector store. No API keys required.

### For New Contributors:

1. **Install Ollama:**
   - Visit https://ollama.com and download for your OS
   - Verify: `ollama --version`

2. **Pull required models:**
   ```bash
   ollama pull qwen2.5
   ollama pull nomic-embed-text
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Copy the example environment file:**
   ```bash
   cd backend
   cp .env.example .env
   ```

5. **Verify your setup:**
   ```bash
   # Check Ollama is running
   curl http://localhost:11434/api/tags
   ```

### Security Notes

- ✅ `.env` is already in `.gitignore` - it will NOT be committed
- ✅ No API keys needed — all inference runs locally via Ollama
- ✅ Vector store persisted locally in `.vector_store.json`

### Troubleshooting

**Ollama not responding?**
- Make sure `ollama serve` is running in a terminal
- Check it’s on port 11434: `curl http://localhost:11434/api/tags`

**Models missing?**
- Run: `ollama pull qwen2.5 && ollama pull nomic-embed-text`

**`.env` file doesn't exist?**
- Run `cp .env.example .env` from the `backend/` directory
