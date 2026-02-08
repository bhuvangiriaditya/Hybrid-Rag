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
- `app.py`: Streamlit app with Chat + Evaluation pages (shared index)
- `data_loader.py`: data acquisition and scraping
- `indexer.py`: chunking, BM25, FAISS
- `rag_engine.py`: retrieval, fusion, generation
- `evaluate.py`: batch evaluation runner (JSON output)
- `eval_dashboard.py`: evaluation dashboard (also usable standalone)
- `pipeline.py`: single-command automated pipeline (JSON/CSV/HTML/PDF)
- `generate_questions.py`: question generation script (100 Q&A)
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
Note: The Evaluation page is also available inside `app.py`.

**Run the Automated Pipeline**
```bash
python3 pipeline.py
```
Outputs in `reports/`:
- `evaluation_results.json`
- `evaluation_results.csv`
- `report.html`
- `report.pdf` (requires `reportlab`)
- `plots/` (visualizations)
- `screenshots/` (placeholders or real screenshots)

Optional flags:
- `--no-generate` (skip answer generation)
- `--skip-ablation` (skip ablation grid)
- `--ablation-generate` (enable answer generation during ablation; slow)
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

**Question Generation**
```bash
python3 generate_questions.py
```
Regenerates `data/rag_questions.json` and `data/rag_questions.csv` from `data/scraped_all.json` (falls back to `data/scraped_fixed.json` if needed).

**Data Requirements**
- Fixed 200 Wikipedia URLs: `data/fixed_urls.json`
- Preprocessed corpus (fixed): `data/scraped_fixed.json`
- Preprocessed corpus (random): `data/scraped_random.json`
- Combined corpus (all 500, chunked): `data/scraped_all.json`
- 100-question dataset: `data/rag_questions.json` (or regenerate with `generate_questions.py`)
- 100-question dataset (CSV): `data/rag_questions.csv` (or regenerate with `generate_questions.py`)
- Evaluation results: `reports/evaluation_results.json` (or regenerate with `pipeline.py`)
- Vector index: built at runtime by `indexer.py` (regeneration instructions included)

**Notes**
- The combined corpus is stored in `data/scraped_all.json`.
- `data/rag_questions.json` includes ground-truth answers and source IDs.
