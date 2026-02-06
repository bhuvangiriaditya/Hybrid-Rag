import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

from data_loader import DataLoader
from indexer import Indexer
from rag_engine import RAGEngine

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None
    letter = None


def load_questions(filepath):
    if not os.path.exists(filepath):
        return {"questions": [], "sources": [], "description": ""}
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "questions": data.get("questions", []),
        "sources": data.get("sources", []),
        "description": data.get("description", ""),
    }


def normalize_tokens(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return tokens


def token_f1(prediction, ground_truth):
    pred_tokens = normalize_tokens(prediction)
    truth_tokens = normalize_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def contextual_precision(answer, context_chunks):
    answer_tokens = normalize_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_text = " ".join([c.get("text", "") for c in context_chunks])
    context_tokens = set(normalize_tokens(context_text))
    if not context_tokens:
        return 0.0
    overlap = sum(1 for t in answer_tokens if t in context_tokens)
    return overlap / len(answer_tokens)


def unique_preserve(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def make_paraphrase(question, category):
    if category == "factual" and question.lower().startswith("what is "):
        title = question[8:].rstrip("?")
        return f"Can you briefly define {title}?"
    if category == "comparative" and "Compare " in question:
        return question.replace("Compare ", "How do ").replace(
            " in one sentence each:", ", and what does each describe?"
        )
    if category == "inferential":
        return f"Which concept best matches this description: {question}"
    if category == "multi-hop":
        return question.replace("Using both sources, state", "Using both sources, summarize")
    return f"Paraphrase: {question}"


def make_unanswerable(question, category):
    if category == "factual" and question.lower().startswith("what is "):
        title = question[8:].rstrip("?")
        return f"What is the capital of {title}?"
    if category == "comparative":
        return question.replace("what does each describe", "which has the larger population")
    if category == "inferential":
        return f"Who is the founder of {question}?"
    if category == "multi-hop":
        return f"Which year did {question}?"
    return f"What is the population of {question}?"


def build_error_labels(results, f1_threshold=0.2, cp_threshold=0.2):
    labels = []
    for r in results:
        if r.get("first_correct_url_rank") is None:
            labels.append("retrieval_failure")
        elif r.get("contextual_precision", 0.0) < cp_threshold:
            labels.append("context_failure")
        elif r.get("answer_f1", 0.0) < f1_threshold:
            labels.append("generation_failure")
        else:
            labels.append("success")
    return labels


def retrieve_context(rag, query, mode, top_k, top_n, rrf_k):
    rag.k = rrf_k
    dense_res, dense_idx = rag.dense_retrieval(query, top_k=top_k)
    sparse_res, sparse_idx = rag.sparse_retrieval(query, top_k=top_k)
    if mode == "dense":
        context = dense_res[:top_n]
    elif mode == "sparse":
        context = sparse_res[:top_n]
    else:
        context = rag.rrf_fusion(dense_idx, sparse_idx, top_n=top_n)
    return context, dense_res, sparse_res


def evaluate_questions(questions, sources, mode, top_k, top_n, rrf_k, generate_answers):
    data_loader = DataLoader()
    documents = data_loader.load_all_data()
    indexer = Indexer()
    indexer.build_index(documents)
    rag = RAGEngine(indexer)

    id_to_url = {s.get("id"): s.get("url") for s in sources}
    results = []
    mrr_sum = 0.0
    precision_at_5_sum = 0.0
    f1_sum = 0.0
    cp_sum = 0.0
    distinct_1_tokens = []
    latencies = []

    for q in questions:
        t0 = time.time()
        context, dense_res, sparse_res = retrieve_context(
            rag, q["question"], mode, top_k, top_n, rrf_k
        )
        answer = ""
        if generate_answers:
            answer = rag.generate_answer(q["question"], context)
        latency = time.time() - t0
        latencies.append(latency)

        retrieved_urls = unique_preserve(
            [c.get("url") for c in context if c.get("url")]
        )
        correct_ids = q.get("source_ids", [])
        correct_urls = [id_to_url.get(sid) for sid in correct_ids if id_to_url.get(sid)]

        rank = None
        if correct_urls:
            for idx, url in enumerate(retrieved_urls):
                if url in correct_urls:
                    rank = idx + 1
                    break
        mrr_sum += (1.0 / rank) if rank else 0.0

        k = 5
        top_k_urls = retrieved_urls[:k]
        hits = sum(1 for url in top_k_urls if url in correct_urls) if correct_urls else 0
        precision_at_5_sum += hits / k

        f1 = token_f1(answer, q.get("ground_truth", "")) if generate_answers else 0.0
        f1_sum += f1

        cp = contextual_precision(answer, context) if generate_answers else 0.0
        cp_sum += cp

        distinct_1_tokens.extend(normalize_tokens(answer))

        results.append({
            "question_id": q.get("id"),
            "question": q.get("question"),
            "category": q.get("category"),
            "ground_truth": q.get("ground_truth"),
            "source_ids": q.get("source_ids", []),
            "generated_answer": answer,
            "retrieved_urls": retrieved_urls,
            "first_correct_url_rank": rank,
            "precision_at_5": hits / k,
            "answer_f1": f1,
            "contextual_precision": cp,
            "latency": latency,
            "dense_titles": [c.get("title") for c in dense_res],
            "sparse_titles": [c.get("title") for c in sparse_res],
            "context_titles": [c.get("title") for c in context],
        })

    metrics = {
        "mrr_url": mrr_sum / len(results) if results else 0.0,
        "precision_at_5_url": precision_at_5_sum / len(results) if results else 0.0,
        "answer_f1": f1_sum / len(results) if results else 0.0,
        "contextual_precision": cp_sum / len(results) if results else 0.0,
        "distinct_1": (len(set(distinct_1_tokens)) / len(distinct_1_tokens))
        if distinct_1_tokens else 0.0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    return metrics, results


def run_adversarial(sample, sources, rag, mode, top_k, top_n, rrf_k, generate_answers):
    id_to_url = {s.get("id"): s.get("url") for s in sources}
    paraphrase_hits = 0
    unanswerable_hallucinations = 0
    paraphrase_details = []
    unanswerable_details = []

    for q in sample:
        paraphrased = make_paraphrase(q["question"], q.get("category", ""))
        context, _, _ = retrieve_context(rag, paraphrased, mode, top_k, top_n, rrf_k)
        retrieved_urls = unique_preserve([c.get("url") for c in context if c.get("url")])
        correct_ids = q.get("source_ids", [])
        correct_urls = [id_to_url.get(sid) for sid in correct_ids if id_to_url.get(sid)]
        actual_url = retrieved_urls[0] if retrieved_urls else None
        hit = bool(actual_url and correct_urls and actual_url in correct_urls)
        if hit:
            paraphrase_hits += 1
        paraphrase_details.append({
            "question": q.get("question"),
            "paraphrased_question": paraphrased,
            "intended_urls": correct_urls,
            "actual_url": actual_url,
            "hit": hit,
        })

        unanswerable = make_unanswerable(q["question"], q.get("category", ""))
        ctx_un, _, _ = retrieve_context(rag, unanswerable, mode, top_k, top_n, rrf_k)
        ans_un = rag.generate_answer(unanswerable, ctx_un) if generate_answers else ""
        cp_un = contextual_precision(ans_un, ctx_un) if generate_answers else 0.0
        hallucinated = bool(ans_un.strip() and cp_un < 0.2)
        if hallucinated:
            unanswerable_hallucinations += 1
        unanswerable_details.append({
            "question": q.get("question"),
            "unanswerable_question": unanswerable,
            "answer": ans_un,
            "contextual_precision": cp_un,
            "hallucinated": hallucinated,
        })

    total = len(sample) if sample else 1
    return {
        "paraphrase_hit_rate": paraphrase_hits / total,
        "unanswerable_hallucination_rate": unanswerable_hallucinations / total,
        "paraphrase_details": paraphrase_details,
        "unanswerable_details": unanswerable_details,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_html_report(path, summary):
    metrics = summary.get("metrics", {})
    adv = summary.get("adversarial", {})
    errors = summary.get("error_analysis", {})
    ablation = summary.get("ablation", [])

    html = []
    html.append("<html><head><meta charset='utf-8'><style>")
    html.append("body{font-family:Arial, sans-serif; margin:24px;} h1{margin-bottom:8px;} table{border-collapse:collapse; width:100%; margin:12px 0;} th,td{border:1px solid #ccc; padding:6px;} th{background:#f2f2f2;}")
    html.append(".section{margin-top:24px;} .small{color:#666; font-size:12px;}")
    html.append("</style></head><body>")
    html.append("<h1>RAG Evaluation Report</h1>")

    html.append("<div class='section'><h2>Core Metrics</h2><table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    for k, v in metrics.items():
        html.append(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>")
    html.append("</table></div>")

    html.append("<div class='section'><h2>Adversarial Testing</h2><table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    html.append(f"<tr><td>paraphrase_hit_rate</td><td>{adv.get('paraphrase_hit_rate', 0.0):.4f}</td></tr>")
    html.append(f"<tr><td>unanswerable_hallucination_rate</td><td>{adv.get('unanswerable_hallucination_rate', 0.0):.4f}</td></tr>")
    html.append("</table><div class='small'>Details included in JSON output.</div></div>")

    html.append("<div class='section'><h2>Error Analysis</h2><table>")
    html.append("<tr><th>Category</th><th>Counts</th></tr>")
    for cat, counts in errors.items():
        html.append(f"<tr><td>{cat}</td><td>{counts}</td></tr>")
    html.append("</table></div>")

    if ablation:
        html.append("<div class='section'><h2>Ablation Grid (Summary)</h2><table>")
        html.append("<tr><th>mode</th><th>top_k</th><th>top_n</th><th>rrf_k</th><th>mrr_url</th><th>precision_at_5_url</th><th>answer_f1</th><th>contextual_precision</th></tr>")
        for row in ablation:
            html.append(
                "<tr>"
                f"<td>{row['mode']}</td>"
                f"<td>{row['top_k']}</td>"
                f"<td>{row['top_n']}</td>"
                f"<td>{row['rrf_k']}</td>"
                f"<td>{row['mrr_url']:.4f}</td>"
                f"<td>{row['precision_at_5_url']:.4f}</td>"
                f"<td>{row['answer_f1']:.4f}</td>"
                f"<td>{row['contextual_precision']:.4f}</td>"
                "</tr>"
            )
        html.append("</table></div>")

    html.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(html))


def write_pdf_report(path, summary):
    if canvas is None:
        raise RuntimeError("reportlab is not installed. Install it to generate PDF reports.")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 48

    def line(text, dy=14, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.drawString(48, y, text)
        y -= dy
        if y < 72:
            c.showPage()
            y = height - 48

    line("RAG Evaluation Report", bold=True, dy=18)
    line("Core Metrics", bold=True)
    for k, v in summary.get("metrics", {}).items():
        line(f"{k}: {v:.4f}")
    line("")
    line("Adversarial Testing", bold=True)
    adv = summary.get("adversarial", {})
    line(f"paraphrase_hit_rate: {adv.get('paraphrase_hit_rate', 0.0):.4f}")
    line(f"unanswerable_hallucination_rate: {adv.get('unanswerable_hallucination_rate', 0.0):.4f}")
    line("")
    line("Error Analysis", bold=True)
    for cat, counts in summary.get("error_analysis", {}).items():
        line(f"{cat}: {counts}")

    c.save()


def main():
    parser = argparse.ArgumentParser(description="Automated RAG evaluation pipeline.")
    parser.add_argument("--questions", default="data/rag_questions.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation.")
    parser.add_argument("--adversarial-sample", type=int, default=20)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    qa_data = load_questions(args.questions)
    questions = qa_data.get("questions", [])
    sources = qa_data.get("sources", [])
    if not questions:
        raise SystemExit("No questions found.")

    metrics, results = evaluate_questions(
        questions, sources, args.mode, args.top_k, args.top_n, args.rrf_k, not args.no_generate
    )

    # Adversarial tests
    data_loader = DataLoader()
    documents = data_loader.load_all_data()
    indexer = Indexer()
    indexer.build_index(documents)
    rag = RAGEngine(indexer)

    sample_n = min(args.adversarial_sample, len(questions))
    sample = random.sample(questions, sample_n)
    adversarial = run_adversarial(
        sample, sources, rag, args.mode, args.top_k, args.top_n, args.rrf_k, not args.no_generate
    )

    # Error analysis
    labels = build_error_labels(results)
    by_cat = defaultdict(Counter)
    for r, label in zip(results, labels):
        by_cat[r.get("category", "unknown")][label] += 1
    error_analysis = {k: dict(v) for k, v in by_cat.items()}

    # Optional ablation
    ablation = []
    if args.ablation:
        grid_modes = ["dense", "sparse", "hybrid"]
        grid_k = [3, 5, 8]
        grid_n = [3, 5]
        grid_rrf = [30, 60]
        for m in grid_modes:
            for k in grid_k:
                for n in grid_n:
                    for rk in grid_rrf:
                        met, _ = evaluate_questions(
                            questions, sources, m, k, n, rk, not args.no_generate
                        )
                        ablation.append({
                            "mode": m,
                            "top_k": k,
                            "top_n": n,
                            "rrf_k": rk,
                            "mrr_url": met.get("mrr_url", 0.0),
                            "precision_at_5_url": met.get("precision_at_5_url", 0.0),
                            "answer_f1": met.get("answer_f1", 0.0),
                            "contextual_precision": met.get("contextual_precision", 0.0),
                        })

    summary = {
        "metadata": {
            "question_source": qa_data.get("description", ""),
            "sources": sources,
            "mode": args.mode,
            "top_k": args.top_k,
            "top_n": args.top_n,
            "rrf_k": args.rrf_k,
            "generate_answers": not args.no_generate,
        },
        "metrics": metrics,
        "adversarial": {
            "paraphrase_hit_rate": adversarial.get("paraphrase_hit_rate", 0.0),
            "unanswerable_hallucination_rate": adversarial.get("unanswerable_hallucination_rate", 0.0),
            "paraphrase_details": adversarial.get("paraphrase_details", []),
            "unanswerable_details": adversarial.get("unanswerable_details", []),
        },
        "error_analysis": error_analysis,
        "ablation": ablation,
        "details": results,
    }

    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    write_csv(csv_path, results)

    html_path = os.path.join(args.output_dir, "report.html")
    write_html_report(html_path, summary)

    pdf_path = os.path.join(args.output_dir, "report.pdf")
    write_pdf_report(pdf_path, summary)

    print("Pipeline complete.")
    print(f"- JSON: {json_path}")
    print(f"- CSV:  {csv_path}")
    print(f"- HTML: {html_path}")
    print(f"- PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
