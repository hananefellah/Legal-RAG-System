# 🏛️ LexAI - Intelligent Legal Research Assistant
### RAG System with LLM Fine-tuning & MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Mistral](https://img.shields.io/badge/Mistral--7B-QLoRA-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

Legal research is time-consuming and complex. **LexAI** is an AI-powered assistant that answers legal questions instantly from a corpus of 500+ legal documents, with source citations, hybrid search, and a fine-tuned LLM.

> **Dataset:** 500+ legal documents, 23,562 indexed chunks  
> **Domain:** Legal NLP / Retrieval-Augmented Generation  
> **Model:** Mistral-7B fine-tuned with QLoRA  

---

## 📊 Results

| Metric | Before Fine-tuning | After Fine-tuning |
|--------|:---:|:---:|
| Faithfulness | 0.575 | **0.840** |
| Context Precision | Baseline | **+90%** |
| Avg Response Time | — | **1.26s** |
| Success Rate | — | **100%** |

---

## 🧠 Key Technical Decisions

### ✅ Hybrid Search (FAISS + BM25 + RRF)
Combined dense vector search (FAISS) with sparse keyword 
search (BM25) using Reciprocal Rank Fusion - gets the 
best of both worlds.

### ✅ Cross-Encoder Reranking
Top 10 candidates reranked with 
`cross-encoder/ms-marco-MiniLM-L-6-v2` before 
passing to LLM - improves answer quality significantly.

### ✅ QLoRA Fine-tuning
Fine-tuned Mistral-7B on 200 legal QA pairs using 
4-bit quantization - pushed faithfulness from 
0.575 → 0.840 with minimal compute.

### ✅ Relevance Gating
Two-stage pipeline: relevance check before generation 
— system refuses out-of-domain questions instead of 
hallucinating.

### ✅ RAGAS Evaluation
Evaluated with 200 QA pairs across Faithfulness, 
Context Precision, Context Recall, and Answer Relevancy.

---

## 🔍 RAG Pipeline
```
Legal Documents (500+)
↓
Text Chunking (256 chars, overlap 25)
↓
Embeddings (all-MiniLM-L6-v2)
↓
FAISS Index + BM25
↓
Hybrid Search + RRF
↓
Cross-Encoder Reranking
↓
Relevance Check
↓
Mistral-7B (fine-tuned) → Answer + Sources
```
---

## 📁 Project Structure

```
Legal-RAG-System/
├── src/
│   └── serving/
│       └── api.py              # FastAPI backend
├── frontend/
│   ├── index.html              # Chat interface
│   ├── script.js               # Frontend logic
│   └── style.css               # Styling
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_evaluation.ipynb
│   ├── 03_retrieval_optimization.ipynb
│   ├── 04_hybrid_search.ipynb
│   ├── 05_query_transformation.ipynb
│   ├── 06_finetuning_data_prep.ipynb
│   ├── 07_finetuning.ipynb
│   ├── 08_model_evaluation.ipynb
│   └── 09_monitoring.ipynb
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci_cd.yml               # GitHub Actions
├── .gitignore
└── README.md
```
---

## ⚙️ How to Run

```bash
# 1. Clone
git clone https://github.com/hananefellah/Legal-RAG-System
cd Legal-RAG-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add environment variables
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Start the API
uvicorn src.serving.api:app --reload --port 8000

# 5. Open frontend
open frontend/index.html
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG orchestration |
| FAISS | Vector similarity search |
| BM25 | Keyword search |
| Mistral-7B + QLoRA | Fine-tuned LLM |
| HuggingFace PEFT | LoRA fine-tuning |
| FastAPI | Production API |
| Docker | Containerization |
| MLflow | Experiment tracking |
| DVC | Data versioning |
| GitHub Actions | CI/CD pipeline |
| RAGAS | RAG evaluation framework |

---

## 🚀 Live Demo
🌐 [LexAI Live](https://willowy-valkyrie-36e039.netlify.app/)

---

## 🔜 Future Work

- [ ] **Fitness domain adaptation** - swap legal docs for fitness content
- [ ] **Subscription system** - Stripe integration
- [ ] **Cloud deployment** - Render + Netlify full deployment
- [ ] **Multi-language support** - Arabic + French legal documents

---

## 📄 License
*MIT License*

## 👩‍💻 Author

**Fellah Hanane** — Data Scientist  
🌐 [GitHub](https://github.com/hananefellah) · Open to Remote Roles

📧 Email: hananefellah35@gmail.com
