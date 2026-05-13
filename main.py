import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. OpenAI-compatible chat completions and native Ollama endpoints.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")

# ============ REQUEST MODELS ============

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class PullModelRequest(BaseModel):
    name: str

# ============ HEALTH & INFO ============

@app.get("/")
def root():
    return {
        "service": "Ollama FastAPI Server",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "web_ui": "/ui",
        "api_docs": "/api-docs",
        "health": "/health",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "generate": "/api/generate",
            "models": "/api/models",
            "pull": "/api/pull"
        }
    }

@app.get("/health")
def health():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return {"status": "ok", "ollama": "connected"}
    except:
        return {"status": "degraded", "ollama": "not ready"}

# ============ OPENAI-COMPATIBLE ENDPOINTS ============

@app.get("/v1/models")
def list_models():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
        data = r.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            })
        return {"object": "list", "data": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=req.stream, timeout=120)

        if req.stream:
            from fastapi.responses import StreamingResponse
            def streamer():
                for line in r.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")

        data = r.json()
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": data.get("message", {}).get("content", "")
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

# ============ OLLAMA NATIVE ENDPOINTS ============

@app.post("/api/generate")
def generate(req: GenerateRequest):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model,
            "prompt": req.prompt,
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=req.stream, timeout=120)

        if req.stream:
            from fastapi.responses import StreamingResponse
            def streamer():
                for line in r.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.get("/api/models")
def ollama_models():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/pull")
def pull_model(req: PullModelRequest):
    try:
        payload = {"name": req.name, "stream": False}
        r = requests.post(f"{OLLAMA_HOST}/api/pull", json=payload, timeout=300)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

# ============ WEB UI ============

@app.get("/ui", response_class=HTMLResponse)
def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama API Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:40px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{color:#00d4ff;margin-bottom:16px;font-size:1.2rem}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.status{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:600}
.status.ok{background:#1a472a;color:#7ee787}
.status.warn{background:#4a3a1a;color:#ffa500}
.chat-box{height:400px;overflow-y:auto;background:#0f0f1a;border-radius:8px;padding:16px;margin-bottom:16px}
.message{margin-bottom:12px;padding:12px;border-radius:8px;max-width:80%}
.message.user{background:#1a3a5c;margin-left:auto}
.message.assistant{background:#2a2a4a}
.input-row{display:flex;gap:10px}
input{flex:1;background:#0f0f1a;border:1px solid #2a2a4a;color:#e0e0e0;padding:12px;border-radius:8px;font-size:1rem}
button{background:linear-gradient(90deg,#00d4ff,#7b2cbf);color:white;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:1rem}
button:hover{opacity:.9}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.nav{display:flex;gap:10px;margin-bottom:30px}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<div class="nav">
<a href="/ui">Dashboard</a>
<a href="/api-docs">API Docs</a>
<a href="/docs">Swagger UI</a>
<a href="/redoc">ReDoc</a>
</div>
<h1>Ollama API Server</h1>
<p class="subtitle">Self-hosted LLM with FastAPI + Ollama</p>
<div class="card">
<h2>Connection Status</h2>
<div id="status">Checking...</div>
<div style="margin-top:10px;font-size:.9rem;color:#888">Base URL: <span id="baseUrl"></span></div>
</div>
<div class="card">
<h2>API Endpoints</h2>
<div class="endpoint"><span class="method">GET</span><span class="url">/health</span> - Health check</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span> - List models</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span> - Chat (OpenAI-compatible)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span> - Generate text</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/api/models</span> - List models (native)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/pull</span> - Pull model</div>
</div>
<div class="card">
<h2>Test Chat</h2>
<div class="chat-box" id="chatBox"></div>
<div class="input-row">
<input type="text" id="chatInput" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendChat()">
<button onclick="sendChat()">Send</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>Python Example</h2>
<div class="code-block">
import requests<br><br>
url = "<span class='base-url'></span>/v1/chat/completions"<br>
headers = {"Content-Type":"application/json"}<br>
data = {"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}<br><br>
res = requests.post(url, json=data, headers=headers)<br>
print(res.json())
</div>
</div>
<div class="card">
<h2>cURL Example</h2>
<div class="code-block">
curl -X POST <span class='base-url'></span>/v1/chat/completions<br>
-H "Content-Type: application/json"<br>
-d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}'
</div>
</div>
</div>
</div>
<script>
const baseUrl = window.location.origin;
document.getElementById('baseUrl').textContent = baseUrl;
document.querySelectorAll('.base-url').forEach(el=>el.textContent=baseUrl);
async function checkHealth(){
try{
const res = await fetch(baseUrl + '/health');
const data = await res.json();
const el = document.getElementById('status');
if(data.ollama === 'connected'){
el.innerHTML = '<span class="status ok">Ollama Connected</span>';
}else{
el.innerHTML = '<span class="status warn">Ollama Starting...</span>';
}
}catch{
document.getElementById('status').innerHTML = '<span class="status warn">Server Starting...</span>';
}
}
checkHealth();
setInterval(checkHealth, 5000);
async function sendChat(){
const input = document.getElementById('chatInput');
const box = document.getElementById('chatBox');
const msg = input.value.trim();
if(!msg) return;
box.innerHTML += `<div class="message user">${msg}</div>`;
input.value = '';
box.scrollTop = box.scrollHeight;
try{
const res = await fetch(baseUrl + '/v1/chat/completions', {
method: 'POST',
headers: {'Content-Type': 'application/json'},
body: JSON.stringify({model: 'qwen2.5:0.5b', messages: [{role: 'user', content: msg}]})
});
const data = await res.json();
const reply = data.choices?.[0]?.message?.content || 'No response';
box.innerHTML += `<div class="message assistant">${reply}</div>`;
box.scrollTop = box.scrollHeight;
}catch(e){
box.innerHTML += `<div class="message assistant" style="color:#ff6b6b">Error: ${e.message}</div>`;
}
}
</script>
</body>
</html>"""

@app.get("/api-docs", response_class=HTMLResponse)
def api_docs():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Documentation - Ollama Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:1000px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:40px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{color:#00d4ff;margin-bottom:16px;font-size:1.2rem}
.card h3{color:#e0e0e0;margin:20px 0 10px;font-size:1rem}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.param{color:#ffa500}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
.table{width:100%;border-collapse:collapse;margin:10px 0}
.table th,.table td{padding:10px;text-align:left;border-bottom:1px solid #2a2a4a;font-size:.9rem}
.table th{color:#00d4ff;font-weight:600}
.table td{color:#ccc}
.nav{display:flex;gap:10px;margin-bottom:30px}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
.note{background:#1a3a5c;border-left:3px solid #00d4ff;padding:12px 16px;margin:10px 0;border-radius:0 8px 8px 0;font-size:.9rem;color:#e0e0e0}
</style>
</head>
<body>
<div class="container">
<div class="nav">
<a href="/ui">Dashboard</a>
<a href="/api-docs">API Docs</a>
<a href="/docs">Swagger UI</a>
<a href="/redoc">ReDoc</a>
</div>
<h1>API Documentation</h1>
<p class="subtitle">Complete reference for Ollama FastAPI Server endpoints</p>

<div class="card">
<h2>Base URL</h2>
<div class="code-block" id="baseUrl">https://your-app.railway.app</div>
<div class="note">All endpoints are relative to your deployment base URL. CORS is enabled for all origins.</div>
</div>

<div class="card">
<h2>Authentication</h2>
<p style="color:#888;margin-bottom:10px">No authentication required by default. Add your own auth middleware if needed.</p>
</div>

<div class="card">
<h2>OpenAI-Compatible Endpoints</h2>
<h3>List Models</h3>
<div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span></div>
<p style="color:#888;margin:8px 0">Returns a list of available models in OpenAI format.</p>
<div class="code-block">Response:<br>
{<br>
&nbsp;&nbsp;"object": "list",<br>
&nbsp;&nbsp;"data": [<br>
&nbsp;&nbsp;&nbsp;&nbsp;{<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"id": "qwen2.5:0.5b",<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"object": "model",<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"created": 1234567890,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"owned_by": "ollama"<br>
&nbsp;&nbsp;&nbsp;&nbsp;}<br>
&nbsp;&nbsp;]<br>
}</div>

<h3>Chat Completions</h3>
<div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span></div>
<p style="color:#888;margin:8px 0">OpenAI-compatible chat endpoint. Supports streaming.</p>
<table class="table">
<tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td>model</td><td>string</td><td>No</td><td>Model name (default: qwen2.5:0.5b)</td></tr>
<tr><td>messages</td><td>array</td><td>Yes</td><td>List of {role, content} objects</td></tr>
<tr><td>stream</td><td>boolean</td><td>No</td><td>Stream response (default: false)</td></tr>
<tr><td>temperature</td><td>float</td><td>No</td><td>Sampling temperature (default: 0.7)</td></tr>
<tr><td>max_tokens</td><td>integer</td><td>No</td><td>Max tokens to generate (default: 2048)</td></tr>
</table>
<div class="code-block">Request:<br>
curl -X POST <span class="base-url"></span>/v1/chat/completions <br>
-H "Content-Type: application/json" <br>
-d '{<br>
&nbsp;&nbsp;"model": "qwen2.5:0.5b",<br>
&nbsp;&nbsp;"messages": [{"role": "user", "content": "Hello!"}],<br>
&nbsp;&nbsp;"temperature": 0.7,<br>
&nbsp;&nbsp;"max_tokens": 2048<br>
}'</div>
</div>

<div class="card">
<h2>Ollama Native Endpoints</h2>
<h3>Generate Text</h3>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span></div>
<table class="table">
<tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td>model</td><td>string</td><td>No</td><td>Model name (default: qwen2.5:0.5b)</td></tr>
<tr><td>prompt</td><td>string</td><td>Yes</td><td>Text prompt</td></tr>
<tr><td>stream</td><td>boolean</td><td>No</td><td>Stream response</td></tr>
<tr><td>temperature</td><td>float</td><td>No</td><td>Sampling temperature</td></tr>
<tr><td>max_tokens</td><td>integer</td><td>No</td><td>Max tokens</td></tr>
</table>

<h3>List Models (Native)</h3>
<div class="endpoint"><span class="method">GET</span><span class="url">/api/models</span></div>
<p style="color:#888;margin:8px 0">Returns models in Ollama native format.</p>

<h3>Pull Model</h3>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/pull</span></div>
<table class="table">
<tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td>name</td><td>string</td><td>Yes</td><td>Model name to pull (e.g., "llama3.2:1b")</td></tr>
</table>
<div class="code-block">curl -X POST <span class="base-url"></span>/api/pull <br>
-H "Content-Type: application/json" <br>
-d '{"name": "llama3.2:1b"}'</div>
</div>

<div class="card">
<h2>Health & Status</h2>
<div class="endpoint"><span class="method">GET</span><span class="url">/health</span></div>
<p style="color:#888;margin:8px 0">Returns server and Ollama connection status.</p>
<div class="code-block">{<br>
&nbsp;&nbsp;"status": "ok",<br>
&nbsp;&nbsp;"ollama": "connected"<br>
}</div>
</div>

<div class="card">
<h2>Environment Variables</h2>
<table class="table">
<tr><th>Variable</th><th>Default</th><th>Description</th></tr>
<tr><td>DEFAULT_MODEL</td><td>qwen2.5:0.5b</td><td>Model auto-pulled on startup</td></tr>
<tr><td>OLLAMA_HOST</td><td>http://localhost:11434</td><td>Internal Ollama server URL</td></tr>
</table>
</div>

<div class="card">
<h2>Available Models</h2>
<table class="table">
<tr><th>Model</th><th>Size</th><th>Speed</th><th>Use Case</th></tr>
<tr><td>qwen2.5:0.5b</td><td>~300MB</td><td>Very Fast</td><td>Default, lightweight tasks</td></tr>
<tr><td>llama3.2:1b</td><td>~1.3GB</td><td>Fast</td><td>General purpose</td></tr>
<tr><td>gemma2:2b</td><td>~1.6GB</td><td>Fast</td><td>Google model, good quality</td></tr>
<tr><td>phi3:mini</td><td>~2GB</td><td>Medium</td><td>Microsoft model, balanced</td></tr>
<tr><td>mistral:7b</td><td>~4GB</td><td>Slower</td><td>High quality, needs more RAM</td></tr>
</table>
<div class="note">To use a different model, set the DEFAULT_MODEL env var or use the /api/pull endpoint.</div>
</div>

</div>
<script>
document.querySelectorAll('.base-url').forEach(el=>el.textContent=window.location.origin);
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
