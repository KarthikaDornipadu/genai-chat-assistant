import os
import sys
import json
import math
import time

# Force unbuffered output
def log(msg):
    print(f"[APP LOG] {msg}", flush=True)

log(f"Initializing app on Python {sys.version}")

from flask import Flask, request, jsonify, render_template

# 1. Initialize App First
app = Flask(__name__)

# 2. Lazy Import GenAI to prevent startup import crashes
genai = None

def get_genai():
    global genai
    if genai is None:
        import google.generativeai as g
        genai = g
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            log("Gemini API configured successfully.")
        else:
            log("CRITICAL: GEMINI_API_KEY is not set.")
    return genai

# 3. State
DOCS_PATH = "docs.json"
chunks = []
chunk_embeddings = []
conversation_history = {}

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0: return 0.0
    return dot_product / (magnitude1 * magnitude2)

def load_docs_lazy():
    global chunks, chunk_embeddings
    if chunks: return # Already loaded
    
    log("Loading knowledge base documents...")
    try:
        if not os.path.exists(DOCS_PATH):
            log(f"Docs file {DOCS_PATH} missing.")
            return
            
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        g = get_genai()
        for doc in docs:
            text = f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
            chunks.append(text)
            embedding = g.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )['embedding']
            chunk_embeddings.append(embedding)
        log(f"Knowledge base loaded: {len(chunks)} chunks.")
    except Exception as e:
        log(f"Failed to load knowledge base: {e}")

# Routes
@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "python": sys.version,
        "key_set": bool(os.getenv("GEMINI_API_KEY")),
        "chunks": len(chunks)
    })

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    # Load docs on first request to avoid deployment timeout
    load_docs_lazy()
    
    data = request.json
    message = data.get("message")
    session_id = data.get("sessionId", "default-session")

    if not message:
        return jsonify({"reply": "Please enter a message.", "tokensUsed": 0, "retrievedChunks": 0})

    try:
        g = get_genai()
        # 1. Retrieval
        query_emb = g.embed_content(
            model="models/gemini-embedding-001",
            content=message,
            task_type="retrieval_query"
        )['embedding']
        
        sims = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        context = [chunks[i] for i in top_indices if sims[i] >= 0.65]
        
        # 2. History
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        history = conversation_history[session_id][-6:]
        
        # 3. Prompt
        ctx_text = "\n\n".join(context) if context else "No context found."
        prompt = f"Context:\n{ctx_text}\n\nQuestion: {message}\nAssistant:"
        
        model = g.GenerativeModel("gemini-1.5-flash") # 1.5-flash is stable
        response = model.generate_content(prompt)
        reply = response.text

        conversation_history[session_id].append({"role": "user", "parts": [message]})
        conversation_history[session_id].append({"role": "model", "parts": [reply]})

        return jsonify({
            "reply": reply,
            "tokensUsed": getattr(response, 'usage_metadata', {}).get('total_token_count', 0),
            "retrievedChunks": len(context)
        })
    except Exception as e:
        log(f"Chat Error: {e}")
        return jsonify({"reply": "Sorry, I encountered an error. Please try again.", "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
