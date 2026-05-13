#!/bin/bash
set -e
ollama serve &
for i in {1..60}; do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama ready"
    break
  fi
  sleep 1
done
DEFAULT_MODEL=${DEFAULT_MODEL:-qwen2.5:0.5b}
ollama pull $DEFAULT_MODEL || true
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
