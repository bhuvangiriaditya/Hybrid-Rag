import json
import time
import re
from collections import Counter
from data_loader import DataLoader
from indexer import Indexer
from rag_engine import RAGEngine
import os

def load_evaluation_questions(filepath="data/rag_questions.json"):
    """
    Loads questions + metadata from the specified JSON file.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {"questions": [], "sources": [], "description": ""}
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        "questions": data.get("questions", []),
        "sources": data.get("sources", []),
        "description": data.get("description", "")
    }

def _normalize_tokens(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return tokens

def token_f1(prediction, ground_truth):
    pred_tokens = _normalize_tokens(prediction)
    truth_tokens = _normalize_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def unique_preserve(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def run_evaluation():
    print("Initializing System for Evaluation...")
    data_loader = DataLoader()
    indexer = Indexer()
    rag = RAGEngine(indexer)

    # 1. Load and Index Data
    print("Loading Data...")
    documents = data_loader.load_all_data()
    print("Indexing Data...")
    indexer.build_index(documents)

    # 2. Load Questions
    print("Loading Questions...")
    qa_data = load_evaluation_questions()
    questions = qa_data.get("questions", [])
    if not questions:
        print("No questions found. Exiting.")
        return

    # 3. Run Pipeline
    results = []
    print(f"Running Evaluation Loop on {len(questions)} questions...")
    
    total_latency = 0
    mrr_sum = 0.0
    precision_at_5_sum = 0.0
    f1_sum = 0.0

    sources = qa_data.get("sources", [])
    id_to_url = {s.get("id"): s.get("url") for s in sources}
    
    for i, q in enumerate(questions):
        print(f"Processing {i+1}/{len(questions)}: {q['question']}")
        start_t = time.time()
        res = rag.process_query(q['question'])
        latency = time.time() - start_t
        total_latency += latency
        
        retrieved_titles = [c['title'] for c in res.get('context', [])]
        retrieved_urls = [c.get('url') for c in res.get('context', []) if c.get('url')]
        retrieved_urls_unique = unique_preserve(retrieved_urls)

        # URL-level MRR and Precision@5
        correct_ids = q.get('source_ids', [])
        correct_urls = [id_to_url.get(sid) for sid in correct_ids if id_to_url.get(sid)]
        rank = None
        if correct_urls:
            for idx, url in enumerate(retrieved_urls_unique):
                if url in correct_urls:
                    rank = idx + 1
                    break
        mrr_sum += (1.0 / rank) if rank else 0.0

        k = 5
        top_k_urls = retrieved_urls_unique[:k]
        if correct_urls:
            hits = sum(1 for url in top_k_urls if url in correct_urls)
            precision_at_5_sum += hits / k
        else:
            precision_at_5_sum += 0.0

        # Answer F1
        f1_sum += token_f1(res.get('answer', ''), q.get('ground_truth', ''))
        
        results.append({
            "question_id": q.get('id'),
            "question": q.get('question'),
            "ground_truth": q.get('ground_truth'),
            "category": q.get('category'),
            "source_ids": q.get('source_ids', []),
            "generated_answer": res.get('answer', "No answer generated"),
            "retrieved_titles": retrieved_titles,
            "retrieved_urls": retrieved_urls_unique,
            "first_correct_url_rank": rank,
            "latency": latency
        })
    
    # 4. Metrics
    avg_latency = total_latency / len(results) if results else 0
    mrr_url = mrr_sum / len(results) if results else 0.0
    precision_at_5_url = precision_at_5_sum / len(results) if results else 0.0
    answer_f1 = f1_sum / len(results) if results else 0.0

    print("\nXXX Evaluation Results XXX")
    print(f"Total Questions: {len(results)}")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"MRR (URL-level): {mrr_url:.4f}")
    print(f"Precision@5 (URL-level): {precision_at_5_url:.4f}")
    print(f"Answer F1: {answer_f1:.4f}")
    
    eval_output_file = "evaluation_results.json"
    with open(eval_output_file, "w") as f:
        json.dump({
            "metadata": {
                "question_source": qa_data.get("description", ""),
                "sources": qa_data.get("sources", [])
            },
            "metrics": {
                "avg_latency": avg_latency,
                "mrr_url": mrr_url,
                "precision_at_5_url": precision_at_5_url,
                "answer_f1": answer_f1
            },
            "details": results
        }, f, indent=2)
    print(f"Results saved to {eval_output_file}")

if __name__ == "__main__":
    run_evaluation()
