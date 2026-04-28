import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import faiss
import re
import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────
GROQ_API_KEY = "GROQ_API_KEY"  # paste your key
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

# ── FastAPI App ────────────────────────────────────────
app = FastAPI(
    title="Legal RAG Assistant API",
    description="AI-powered legal document search and Q&A system",
    version="1.0.0"
)
# Add Prometheus monitoring
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ────────────────────────────
class HistoryItem(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    history: Optional[list[HistoryItem]] = []

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    response_time: float

class HealthResponse(BaseModel):
    status: str
    model: str
    chunks_indexed: int

# ── Global variables ───────────────────────────────────
chunks = []
faiss_index = None
embed_model = None
reranker = None
bm25 = None
llm = None
start_time = time.time()

# ── Helper functions ───────────────────────────────────
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def hybrid_search(query, top_k=10, rrf_k=60):
    query_vector = embed_model.encode(
        [query], convert_to_numpy=True
    ).astype(np.float32)
    _, faiss_indices = faiss_index.search(query_vector, 50)
    faiss_ranking = {idx: rank for rank, idx in enumerate(faiss_indices[0])}

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:50]
    bm25_ranking = {idx: rank for rank, idx in enumerate(bm25_indices)}

    all_indices = set(faiss_ranking.keys()) | set(bm25_ranking.keys())
    rrf_scores = {}
    for idx in all_indices:
        faiss_score = 1 / (rrf_k + faiss_ranking.get(idx, 1000))
        bm25_score  = 1 / (rrf_k + bm25_ranking.get(idx, 1000))
        rrf_scores[idx] = faiss_score + bm25_score

    top_indices = sorted(
        rrf_scores.keys(),
        key=lambda x: rrf_scores[x],
        reverse=True
    )[:top_k]
    return [{"text": chunks[idx]["text"], "chunk_id": chunks[idx]["chunk_id"]} for idx in top_indices]

def hybrid_search_with_reranking(query, top_k=3):
    candidates = hybrid_search(query, top_k=10)
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )
    return [c for _, c in ranked[:top_k]]

# ── Startup Event ──────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global chunks, faiss_index, embed_model, reranker, bm25, llm

    print("⏳ Loading RAG system...")

    # Load chunks
    with open(os.path.join(DATA_DIR, "processed/chunks.json"), "r") as f:
        all_chunks = json.load(f)

    # Use 256-char chunks (Week 3 winner)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=25,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    with open(os.path.join(DATA_DIR, "raw/legal_documents.json"), "r") as f:
        documents = json.load(f)

    for doc in documents:
        doc_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{i:03d}",
                "doc_id": doc["doc_id"],
                "text": chunk_text,
            })

    # Load embedding model
    print("⏳ Loading models...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Check if embeddings already exist
    embeddings_path = os.path.join(DATA_DIR, "embeddings/embeddings_256.npy")
    if os.path.exists(embeddings_path):
        print("✅ Loading saved embeddings from disk...")
        embeddings = np.load(embeddings_path)
    else:
        print("⏳ Generating embeddings (first time only)...")
        embeddings = embed_model.encode(
            [c["text"] for c in chunks],
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        np.save(embeddings_path, embeddings)
        print("✅ Embeddings saved to disk!")

    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype(np.float32))

    # Build BM25
    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Load LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    print(f"✅ RAG system ready! {len(chunks)} chunks indexed")

# ── API Endpoints ──────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model="llama-3.1-8b-instant",
        chunks_indexed=len(chunks)
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        start = time.time()

        # Retrieve relevant chunks
        relevant_chunks = hybrid_search_with_reranking(
            request.question, top_k=request.top_k
        )
        context_text = "\n\n".join([c["text"] for c in relevant_chunks])

        # Generate answer
        # Build conversation history for context
        history_text = ""
        if request.history:
            # Keep last 6 messages (3 exchanges) to avoid token overflow
            recent_history = request.history[-6:]
            for msg in recent_history:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role_label}: {msg.content}\n"
            history_text = f"\nConversation History:\n{history_text}\n"

        # Step 1: Relevance check — does the context actually answer this question?
        relevance_prompt = f"""Decide if the following context contains information RELATED to the question.
The context does not need a perfect definition — if it discusses the topic, mentions relevant laws, 
penalties, or concepts related to the question, answer YES.
Only answer NO if the context is completely unrelated to the question.

Context: {context_text}

Question: {request.question}

Reply with ONLY "YES" or "NO". Nothing else."""

        relevance_check = llm.invoke(relevance_prompt)
        is_relevant = relevance_check.content.strip().upper().startswith("YES")

        if not is_relevant:
            answer = "I can only answer questions related to the legal documents in my knowledge base. Please ask a legal-related question."
        else:
            # Step 2: Generate answer ONLY from context
            prompt = f"""You are a legal assistant. Answer the question using ONLY the context below.

Rules:
- Use ONLY information explicitly stated in the context. Do NOT add outside knowledge.
- Keep your answer concise: 3-5 sentences maximum.
- If the context only partially answers the question, answer only the part you can support with the context.
{history_text}
Context: {context_text}

Question: {request.question}

Answer:"""

            response = llm.invoke(prompt)
            answer = response.content.strip()
        elapsed = time.time() - start

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=[c["chunk_id"] for c in relevant_chunks],
            response_time=round(elapsed, 2)
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Legal RAG Assistant API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }