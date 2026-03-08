import os
import sys

# Ensure logs are unbuffered for Render
def log(msg):
    print(msg, flush=True)

log(f"Starting GenAI Chat Assistant on Python {sys.version}")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env for local development; no-op on Render
except ImportError:
    pass  # dotenv not installed (production env — vars set natively)

from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import json
import math

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "embeddings_loaded": len(chunk_embeddings) > 0})

# --- RAG Setup (Logic moved to main block for better Render debugging) ---
DOCS_PATH = "docs.json"
chunks = []
chunk_embeddings = []
conversation_history = {} 

def generate_embedding(text, task_type="retrieval_document"):
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set in environment variables")
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    return result['embedding']

def load_and_embed_documents():
    global chunks, chunk_embeddings
    log("Starting document embedding...")
    try:
        if not os.path.exists(DOCS_PATH):
            log(f"Warning: {DOCS_PATH} not found. Skipping initial embedding.")
            return
            
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        for doc in docs:
            chunk_text = f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')}"
            chunks.append(chunk_text)
            embedding = generate_embedding(chunk_text, task_type="retrieval_document")
            chunk_embeddings.append(embedding)
        log(f"Successfully loaded and embedded {len(chunks)} document chunks.")
    except Exception as e:
        log(f"Error during document embedding: {e}")
        log("Continuing startup without embeddings...")

# Configure GenAI and Load Docs
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        log("Gemini API configured.")
        load_and_embed_documents()
    else:
        log("WARNING: GEMINI_API_KEY not set. App will start but Gemini features will fail.")
except Exception as e:
    log(f"CRITICAL ERROR during GenAI configuration: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    session_id = data.get("sessionId", "default-session")

    if not message:
        return jsonify({"reply": "Please enter a question.", "tokensUsed": 0, "retrievedChunks": 0})

    # 1. Retrieve relevant chunks
    retrieved_chunks_count = 0
    context_texts = []
    try:
        query_embedding = generate_embedding(message, task_type="retrieval_query")
        similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
        
        # Get top 3 indices and threshold > 0.5
        scored_chunks = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)
        
        for idx, score in scored_chunks[:3]:
            # Apply similarity threshold (tuned to 0.65 for gemini-embedding-001)
            if score >= 0.65:
                context_texts.append(chunks[idx])
                retrieved_chunks_count += 1
                
    except Exception as e:
        print(f"Retrieval error: {e}")

    # 2. Build history context
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Keep last 3 message pairs (6 messages total)
    history = conversation_history[session_id][-6:]
    
    chat_history = []
    for msg in history:
        chat_history.append({"role": msg["role"], "parts": [msg["parts"][0]]})
    
    # 3. Construct System Prompt
    context_block = "\n\n".join(context_texts) if context_texts else "No specific documents found in the current knowledge base to answer this query."
    
    system_instruction = f"""
You are a friendly, customer-centric Support Assistant for our application.
Your goal is to provide helpful, detailed, and clear answers that are easy for customers to understand.
Answer the user's question based STRICTLY on the following Context. 
Please format your response clearly:
- Start with a warm, polite greeting.
- Provide a detailed yet concise explanation of the solution.
- Use bullet points or numbered lists if there are multiple steps.
- End with a polite closing asking if they need further assistance.
If the Context does not contain the answer, politely apologize and state that you do not have that specific information in your knowledge base.
Do NOT hallucinate or make up answers not found in the Context.

--- DOCUMENT CONTEXT ---
{context_block}
--- END CONTEXT ---
"""
    
    try:
        chat_model = genai.GenerativeModel(
            "gemini-2.5-flash", 
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        
        chat_session = chat_model.start_chat(history=chat_history)
        
        full_message = f"{system_instruction}\n\nUser Question:\n{message}"
        response = chat_session.send_message(full_message)
        reply = response.text
        
        # Save to history
        conversation_history[session_id].append({"role": "user", "parts": [message]})
        conversation_history[session_id].append({"role": "model", "parts": [reply]})
        
        tokens_used = 0
        try:
             # Natively captures token usage in GenAI standard dict
             if hasattr(response, 'usage_metadata'):
                 tokens_used = response.usage_metadata.total_token_count
        except:
             pass

    except Exception as e:
        print(f"Generation error: {e}")
        reply = "Unable to generate response right now. Please try again later."
        tokens_used = 0

    return jsonify({
        "reply": reply,
        "tokensUsed": tokens_used,
        "retrievedChunks": retrieved_chunks_count
    })

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log(f"Starting local server on port {port}...")
    app.run(host="0.0.0.0", port=port)
