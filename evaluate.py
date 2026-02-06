import json
import time
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
    
    for i, q in enumerate(questions):
        print(f"Processing {i+1}/{len(questions)}: {q['question']}")
        start_t = time.time()
        res = rag.process_query(q['question'])
        latency = time.time() - start_t
        total_latency += latency
        
        # We don't have ground truth for these questions, so we track what was retrieved
        retrieved_titles = [c['title'] for c in res.get('context', [])]
        
        results.append({
            "question_id": q.get('id'),
            "question": q.get('question'),
            "ground_truth": q.get('ground_truth'),
            "category": q.get('category'),
            "source_ids": q.get('source_ids', []),
            "generated_answer": res.get('answer', "No answer generated"),
            "retrieved_titles": retrieved_titles,
            "latency": latency
        })
    
    # 4. Metrics
    avg_latency = total_latency / len(results) if results else 0

    print("\nXXX Evaluation Results XXX")
    print(f"Total Questions: {len(results)}")
    print(f"Average Latency: {avg_latency:.4f}s")
    
    eval_output_file = "evaluation_results.json"
    with open(eval_output_file, "w") as f:
        json.dump({
            "metadata": {
                "question_source": qa_data.get("description", ""),
                "sources": qa_data.get("sources", [])
            },
            "metrics": {"avg_latency": avg_latency},
            "details": results
        }, f, indent=2)
    print(f"Results saved to {eval_output_file}")

if __name__ == "__main__":
    run_evaluation()
