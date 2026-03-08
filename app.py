
import os
import sys
import json
import math
import time

# Guaranteed unbuffered logging
def log(msg):
    print(f"[BOOT] {msg}", flush=True)

log(f"Process started. Python: {sys.version}")

try:
    from flask import Flask, request, jsonify, render_template
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    log("Basic imports successful.")
except Exception as e:
    log(f"IMPORT ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

app = Flask(__name__)

# State
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

def load_knowledge_base():
    global chunks, chunk_embeddings
    log("Attempting to load knowledge base...")
    try:
        if not os.path.exists(DOCS_PATH):
            log(f"Notice: {DOCS_PATH} does not exist.")
            return

        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        # Initialize Gemini for embedding
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log("ABORT: GEMINI_API_KEY missing. Cannot embed docs.")
            return
            
        genai.configure(api_key=api_key)
        
        for doc in docs:
            text = f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
            chunks.append(text)
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            chunk_embeddings.append(res['embedding'])
        log(f"Knowledge base loaded: {len(chunks)} chunks.")
    except Exception as e:
        log(f"KNOWLEDGE BASE ERROR: {e}")
        import traceback
        log(traceback.format_exc())

# Lazy-load docs on first request if needed
def ensure_docs_loaded():
    if not chunks:
        load_knowledge_base()

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "python": sys.version,
        "chunks": len(chunks)
    })

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    ensure_docs_loaded()
    
    data = request.json or {}
    message = data.get("message")
    session_id = data.get("sessionId", "default")
    
    if not message:
        return jsonify({"reply": "Message missing."})
        
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"reply": "API Key is missing."})
        genai.configure(api_key=api_key)
        
        # 1. Embed Query
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=message,
            task_type="retrieval_query"
        )
        query_emb = res['embedding']
        
        # 2. Similarity
        sims = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
        matches = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        context = [chunks[i] for i in matches if sims[i] >= 0.65]
        
        # 3. Chat
        if session_id not in conversation_history:
            conversation_history[session_id] = []
            
        ctx_text = "\n\n".join(context) if context else "No direct info found."
        prompt = f"Context: {ctx_text}\n\nUser: {message}\nAssistant:"
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        reply = response.text
        
        conversation_history[session_id].append({"role": "user", "parts": [message]})
        conversation_history[session_id].append({"role": "model", "parts": [reply]})
        
        return jsonify({
            "reply": reply,
            "retrievedChunks": len(context),
            "tokensUsed": getattr(response, 'usage_metadata', {}).get('total_token_count', 0)
        })
    except Exception as e:
        log(f"CHAT ERROR: {e}")
        return jsonify({"reply": "Error occurred.", "debug": str(e)})

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 10000))
        log(f"Starting server on port {port}...")
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        log(f"SERVER CRASH: {e}")
        sys.exit(1)
