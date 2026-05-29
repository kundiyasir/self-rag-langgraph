# Self-RAG Agentic QA System — LangGraph

A production-style **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** pipeline built incrementally across 8 notebooks using LangGraph, LangChain, and OpenAI. The system goes beyond basic RAG by adding agentic decision-making, hallucination verification, usefulness checking, query rewriting, and web search fallback.

---

## Overview

Standard RAG pipelines retrieve documents and generate answers blindly. This system adds **self-reflection** at every stage — the LLM decides whether to retrieve, checks if retrieved docs are relevant, verifies if its own answer is supported by the context, evaluates whether the answer is useful, and rewrites its query if not.

---

## Repository Structure

```
self-rag-langgraph/
├── self_rag_step1.ipynb   # Retrieval decision + FAISS vector store
├── self_rag_step2.ipynb   # Relevance filtering of retrieved docs
├── self_rag_step3.ipynb   # RAG generation + no-docs fallback
├── self_rag_step4.ipynb   # IsSUP: hallucination verification
├── self_rag_step5.ipynb   # Retry loop + answer revision
├── self_rag_step6.ipynb   # IsUSE: answer usefulness check
├── self_rag_step7.ipynb   # Query rewriting for better retrieval
├── self_rag_web.ipynb     # Web search fallback via Tavily
└── README.md
```

---

## Pipeline Architecture

```
Question
   │
   ▼
[Decide Retrieval] ──── No retrieval needed ──► [Generate Direct] ──► END
   │
   │ Retrieval needed
   ▼
[Retrieve] (FAISS vector store)
   │
   ▼
[Is Relevant?] ── filter irrelevant docs
   │
   ├── No relevant docs ──► [Rewrite Query] ──► [Web Search (Tavily)] ──► END
   │
   ▼
[Generate from Context]
   │
   ▼
[IsSUP — Hallucination Check]
   │
   ├── fully_supported ──► [IsUSE — Usefulness Check]
   │                              │
   ├── partially_supported ──►   ├── useful ──► END
   │        │                    │
   └── no_support ──► [Revise]   └── not_useful ──► [Rewrite Query] ──► loop
              │
              └──► loop back to IsSUP (with retry limit)
```

---

## Features Built Step-by-Step

| Notebook | Feature Added |
|---|---|
| Step 1 | Retrieval decision node · FAISS vector store · LangGraph StateGraph |
| Step 2 | Per-document relevance filtering |
| Step 3 | Context-grounded generation · no-docs fallback node |
| Step 4 | IsSUP verification — `fully_supported / partially_supported / no_support` |
| Step 5 | Retry loop — answer revision when not fully supported |
| Step 6 | IsUSE check — verifies answer addresses the actual question |
| Step 7 | Query rewriting — rewrites question into retrieval-optimized query |
| Web | Tavily web search fallback when internal docs return no relevant results |

---

## Tech Stack

```
Orchestration    LangGraph (StateGraph, conditional edges, retry loops)
LLM Framework    LangChain (ChatGroq, ChatPromptTemplate, structured output)
LLM              Groq — llama-3.1-8b-instant
Embeddings       HuggingFace sentence-transformers (all-MiniLM-L6-v2)
Vector Store     FAISS
Structured IO    Pydantic BaseModel (typed decision nodes)
Web Search       Tavily Search API
Document Load    PyPDFLoader (LangChain Community)
```

---

## Key Design Decisions

- **Pydantic structured outputs** for every decision node — no fragile string parsing
- **Conditional edges** in LangGraph route between nodes based on LLM decisions
- **Recursion-safe retry loops** with explicit `retries` counter in graph state to prevent infinite loops
- **Separate retrieval query** field in state — the rewritten query sent to FAISS is decoupled from the original user question
- **IsSUP before IsUSE** — first verify factual grounding, then check practical usefulness

---

## How to Run

1. Clone the repo
2. Install dependencies:
```bash
pip install langchain langchain-community langchain-groq langgraph faiss-cpu pydantic python-dotenv sentence-transformers langchain-huggingface tavily-python pypdf
```
3. Create a `.env` file:
```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```
   The notebooks load this `.env` automatically via `python-dotenv`. If you run
   them in Google Colab instead, store the same keys as Colab secrets and they
   will be picked up as a fallback.
4. Add your PDF documents to a `documents/` folder
5. Run notebooks in order (step1 → step7 → web)

---

## Author

**Yasir Kundi**
M.S. Applied Statistics, Data Science — West Chester University
[LinkedIn](https://linkedin.com/in/yasir-kundi) · [GitHub](https://github.com/kundiyasir)
