# Ollama FastAPI Server
Self-hosted LLM API using Ollama + FastAPI for Railway.

## Deploy
1. Push to GitHub
2. Railway: New Project -> Deploy from GitHub repo
3. Generate Domain in Railway Settings
4. Done!

## Endpoints
- GET /health
- GET /ui  (Web interface)
- GET /v1/models
- POST /v1/chat/completions (OpenAI-compatible)
- POST /api/generate (Ollama native)
- POST /api/pull

## Env Vars
- DEFAULT_MODEL=qwen2.5:0.5b
- OLLAMA_HOST=http://localhost:11434
