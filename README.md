# Hybrid-RAG

Hybrid RAG system combining dense + sparse retrieval with Reciprocal Rank Fusion, plus evaluation tooling and an automated reporting pipeline.

**Features**
- Dense retrieval with `all-MiniLM-L6-v2` + FAISS
- Sparse retrieval with BM25
- Fusion with Reciprocal Rank Fusion (RRF)
- LLM answer generation with `google/flan-t5-base`
- Evaluation metrics (MRR URL, Precision@5 URL, Answer F1, Contextual Precision, Distinct-1, Latency)
- Adversarial testing (paraphrases, unanswerables)
- Ablation grid (dense vs sparse vs hybrid)
- Streamlit dashboard
- Automated pipeline outputting JSON/CSV/HTML/PDF reports

**Project Layout**
- `app.py`: Streamlit demo app
- `data_loader.py`: data acquisition and scraping
- `indexer.py`: chunking, BM25, FAISS
- `rag_engine.py`: retrieval, fusion, generation
- `evaluate.py`: batch evaluation runner (JSON output)
- `eval_dashboard.py`: interactive evaluation dashboard
- `pipeline.py`: single-command automated pipeline (JSON/CSV/HTML/PDF)
- `data/`: URLs, scraped corpus, and questions

**Setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the App**
```bash
streamlit run app.py
```

**Run Batch Evaluation**
```bash
python3 evaluate.py
```
Outputs: `evaluation_results.json`

**Run the Evaluation Dashboard**
```bash
streamlit run eval_dashboard.py
```

**Run the Automated Pipeline**
```bash
python3 pipeline.py
```
Outputs in `reports/`:
- `evaluation_results.json`
- `evaluation_results.csv`
- `report.html`
- `report.pdf` (requires `reportlab`)

Optional flags:
- `--no-generate` (skip answer generation)
- `--ablation` (run ablation grid)
- `--adversarial-sample N` (default 20)
- `--mode {hybrid,dense,sparse}` (default hybrid)
- `--top-k N`, `--top-n N`, `--rrf-k N`

**Core Metrics (Evaluation)**
- `MRR (URL)`: average of 1/rank for first correct source URL
- `Precision@5 (URL)`: correct URLs in top-5 / 5
- `Answer F1`: token overlap with ground truth
- `Contextual Precision`: fraction of answer tokens found in context
- `Distinct-1`: unique tokens / total tokens
- `Avg Latency`: end-to-end time per question

**Adversarial Testing**
- Paraphrase hit rate: paraphrase query, check if top URL is still correct
- Unanswerable hallucination rate: unanswerable query, count unsupported answers

**Notes**
- The corpus is stored in `data/scraped_fixed.json`.
- `data/rag_questions.json` includes ground-truth answers and source IDs.
