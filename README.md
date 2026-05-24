# 🧠 ResearchMind — AI Research Paper Knowledge Graph

Analyzes research papers using LLMs and builds an interactive knowledge graph showing how ideas connect.

## What it does
- Fetches real research papers from ArXiv / Semantic Scholar
- Uses **Groq LLaMA 3.3 70B** to extract key claims, concepts, and contributions from each paper
- Builds a **knowledge graph** using sentence embeddings and cosine similarity
- Detects **contradictions** between papers
- Interactive **Ask AI** tab — ask questions across all papers, get cited answers

## Features
- 🕸️ Interactive force-directed knowledge graph (pyvis)
- 📄 Per-paper claim extraction with LLM
- 💬 RAG-powered Q&A with citations
- ⚠️ Contradiction detection across papers

## Tech stack
`Python` `Groq API (LLaMA 3.3)` `sentence-transformers` `NetworkX` `pyvis` `Streamlit` `ChromaDB`

## Run locally
```bash
pip install -r requirements.txt
streamlit run 03_dashboard.py
```

## How it works
1. Papers are fetched and their abstracts sent to an LLM for structured extraction
2. Abstracts are embedded using `all-MiniLM-L6-v2`
3. Cosine similarity > 0.5 creates edges between papers
4. Shared concepts create additional concept nodes
5. Q&A uses full paper context injected into the LLM prompt