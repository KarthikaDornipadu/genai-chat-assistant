import os
import sys
import json
import math
import time

# --- 0. FORCE UNBUFFERED LOGS ---
# This ensures we see these messages in Render logs even on a crash
sys.stdout.reconfigure(line_buffering=True)

def diag_log(msg):
    print(f"[RENDER-BOOT-DIAGNOSTIC] {msg}", flush=True)

diag_log(f"Initializing Chat Assistant... Python: {sys.version}")

try:
    from flask import Flask, request, jsonify, render_template
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv() # No-op if .env is missing
    diag_log("Dependencies (Flask, GenAI) loaded successfully.")
except Exception as e:
    diag_log(f"CRITICAL IMPORT ERROR: {e}")
    import traceback
    diag_log(traceback.format_exc())
    sys.exit(1)

# --- 1. INITIALIZE APP IMMEDIATELY ---
# This is crucial so health checks can pass ASAP.
app = Flask(__name__)

# --- 2. STATE ---
DOCS_PATH = os.path.join(os.getcwd(), "docs.json")
chunks = []
chunk_embeddings = []
conversation_history = {}

# --- 3. LOGIC ---
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0: return 0.0
    return dot_product / (magnitude1 * magnitude2)

def bootstrap_knowledge_base():
    global chunks, chunk_embeddings
    if chunks: return # Only load once
    
    diag_log(f"Bootstrapping knowledge base from {DOCS_PATH}...")
    try:
        if not os.path.exists(DOCS_PATH):
            diag_log(f"Warning: {DOCS_PATH} missing. Continuing with empty KB.")
            return

        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            diag_log("Warning: GEMINI_API_KEY not found. KB embedding skipped.")
            return
            
        genai.configure(api_key=api_key)
        
        for doc in docs:
            text = f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
            chunks.append(text)
            # API call for embedding
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            # Adaptive result access (handles different SDK versions)
            emb = res['embedding'] if isinstance(res, dict) else res.embedding
            chunk_embeddings.append(emb)
            
        diag_log(f"Knowledge base ready: {len(chunks)} chunks loaded.")
    except Exception as e:
        diag_log(f"KB INITIALIZATION ERROR: {e}")
        import traceback
        diag_log(traceback.format_exc())

# --- 4. ROUTES ---
@app.route("/health")
def health():
    # Render health check
    return jsonify({"status": "healthy", "python": sys.version, "kb_loaded": len(chunks) > 0})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    # Load KB on the first real request to prevent startup delays
    bootstrap_knowledge_base()
    
    data = request.json or {}
    message = data.get("message")
    session_id = data.get("sessionId", "default")
    
    if not message:
        return jsonify({"reply": "Message is empty."})
        
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"reply": "API Key is not configured on server."})
        genai.configure(api_key=api_key)
        
        # 1. Retrieval
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=message,
            task_type="retrieval_query"
        )
        q_emb = res['embedding'] if isinstance(res, dict) else res.embedding
        
        sims = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
        matches = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        context = [chunks[i] for i in matches if sims[i] >= 0.65]
        
        # 2. History
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # 3. Model
        ctx_text = "\n\n".join(context) if context else "No relevant info found in documentation."
        prompt = f"Context: {ctx_text}\n\nUser: {message}\nAssistant:"
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        reply = response.text
        
        conversation_history[session_id].append({"role": "user", "parts": [message]})
        conversation_history[session_id].append({"role": "model", "parts": [reply]})
        
        return jsonify({
            "reply": reply,
            "retrievedChunks": len(context),
            "tokensUsed": getattr(getattr(response, 'usage_metadata', None), 'total_token_count', 0)
        })
    except Exception as e:
        diag_log(f"RUNTIME CHAT ERROR: {e}")
        return jsonify({"reply": "Oops! I hit a snag. Please try again.", "debug": str(e)})

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # This block runs when executing 'python app.py' locally.
    # On Render, gunicorn will ignore this and use the 'app' object directly.
    port = int(os.environ.get("PORT", 10000))
    diag_log(f"Starting development server on port {port}...")
    app.run(host="0.0.0.0", port=port)
