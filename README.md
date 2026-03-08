# Production-Grade GenAI Assistant with RAG

A production-style GenAI-powered Chat Assistant that answers user questions using Retrieval-Augmented Generation (RAG). Built using Python, Flask, and the Google Gemini API.

## 🏗 Architecture Diagram
```
[User Interface] <---(JSON)---> [Flask Backend /api/chat]
       |                              |
       |  1. Generate Session ID      |  2. Process Query Embedding (gemini-embedding-001)
       |  3. Display Markdown Output  |  4. Search `docs.json` chunks (Cosine Similarity > 0.65)
       |                              |  5. Compile History & Context into Prompt
       |                              |  6. Call LLM (gemini-2.5-flash) with Context
```

## 🧠 RAG Workflow
1. **Startup**: During application launch, the `docs.json` knowledge base is loaded, converted into logical chunks, and processed into high-dimensional vector embeddings stored securely in memory.
2. **Querying**: As a user submits a query to the chat API, the query is similarly vectorized into an embedding representation.
3. **Retrieval**: The system iterates over the knowledge base vectors in memory and employs a native Cosine Similarity math function to select the top 3 chunks bearing a score above `0.65`.
4. **Generation**: The matched relevant contextual chunks, along with the user's conversational history (tracked via a localStorage `sessionId`), are cleanly injected directly into the Gemini completion request payload as explicit instructions to ground the AI response. 
5. **Output**: The generated message is formatted via markdown library seamlessly on the UI.

## 📊 Embedding Strategy
To bypass bloat latency and heavy library usages, this app strictly implements `google.generativeai.embed_content` tied exclusively to `gemini-embedding-001`. The memory array retains embedding arrays for low-millisecond semantic retrieval across documents natively without an overhead vector database server. 

## 🔍 Similarity Search
The backend runs natively calculated **Cosine Similarity** `(dot_product / (magnitude1 * magnitude2))` for performance. It scales up chunks via Python's native list structures. A tuned threshold of `>= 0.65` acts as a hard gate. If a question is entirely separate from the vectors (e.g. "What is my name?"), it scores `< 0.60`, yielding an empty context constraint array so that Gemini immediately returns a fallback instead of hallucinating answers.

## ✍️ Prompt Design
The prompt is meticulously crafted to negate hallucinations and enforce grounding. We use:
- Explicit constraints: `Answer the user's question clearly and simply based STRICTLY on the following Context.`
- Strict formatting brackets `--- DOCUMENT CONTEXT ---`.
- Bounding configurations: We initialize `gemini-2.5-flash` with a strict execution temperature of `0.1` forcing determinism over creative hallucinations. 
- Graceful degradation: `If the Context does not contain the answer, politely state that you do not have enough information.`

## 🚀 Setup Instructions

### 1. Requirements
Ensure Python `3.10+` is installed on your local operating environment.

### 2. Install Packages
Run the following native pip command to procure dependencies:
```bash
pip install flask google-generativeai
```

### 3. Execution
Launch the production flask server. It will automatically structure and cache the document embeddings locally upon generation.
```bash
python app.py
```

Visit the User Interaface at `http://127.0.0.1:5000` to interact with semantic retrieval.
