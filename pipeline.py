import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data_loader import DataLoader
from indexer import Indexer
from rag_engine import RAGEngine

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
except Exception:
    canvas = None
    letter = None
    ImageReader = None


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


def evaluate_questions(questions, sources, rag, mode, top_k, top_n, rrf_k, generate_answers):
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
        reciprocal_rank = (1.0 / rank) if rank else 0.0
        mrr_sum += reciprocal_rank

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
            "reciprocal_rank": reciprocal_rank,
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


def evaluate_hybrid_rrf_list(questions, sources, rag, top_k, top_n, rrf_ks, generate_answers):
    id_to_url = {s.get("id"): s.get("url") for s in sources}
    sums = {}
    distinct_tokens = {k: [] for k in rrf_ks}
    latencies = {k: [] for k in rrf_ks}
    for k in rrf_ks:
        sums[k] = {
            "mrr_sum": 0.0,
            "precision_at_5_sum": 0.0,
            "f1_sum": 0.0,
            "cp_sum": 0.0,
        }

    for q in questions:
        dense_res, dense_idx = rag.dense_retrieval(q["question"], top_k=top_k)
        sparse_res, sparse_idx = rag.sparse_retrieval(q["question"], top_k=top_k)
        for rrf_k in rrf_ks:
            t0 = time.time()
            rag.k = rrf_k
            context = rag.rrf_fusion(dense_idx, sparse_idx, top_n=top_n)
            answer = ""
            if generate_answers:
                answer = rag.generate_answer(q["question"], context)
            latency = time.time() - t0
            latencies[rrf_k].append(latency)

            retrieved_urls = unique_preserve([c.get("url") for c in context if c.get("url")])
            correct_ids = q.get("source_ids", [])
            correct_urls = [id_to_url.get(sid) for sid in correct_ids if id_to_url.get(sid)]

            rank = None
            if correct_urls:
                for idx, url in enumerate(retrieved_urls):
                    if url in correct_urls:
                        rank = idx + 1
                        break
            reciprocal_rank = (1.0 / rank) if rank else 0.0
            sums[rrf_k]["mrr_sum"] += reciprocal_rank

            k = 5
            top_k_urls = retrieved_urls[:k]
            hits = sum(1 for url in top_k_urls if url in correct_urls) if correct_urls else 0
            sums[rrf_k]["precision_at_5_sum"] += hits / k

            f1 = token_f1(answer, q.get("ground_truth", "")) if generate_answers else 0.0
            sums[rrf_k]["f1_sum"] += f1

            cp = contextual_precision(answer, context) if generate_answers else 0.0
            sums[rrf_k]["cp_sum"] += cp

            distinct_tokens[rrf_k].extend(normalize_tokens(answer))

    metrics_by_rrf = {}
    total = len(questions) if questions else 1
    for rrf_k in rrf_ks:
        distinct = distinct_tokens[rrf_k]
        metrics_by_rrf[rrf_k] = {
            "mrr_url": sums[rrf_k]["mrr_sum"] / total,
            "precision_at_5_url": sums[rrf_k]["precision_at_5_sum"] / total,
            "answer_f1": sums[rrf_k]["f1_sum"] / total,
            "contextual_precision": sums[rrf_k]["cp_sum"] / total,
            "distinct_1": (len(set(distinct)) / len(distinct)) if distinct else 0.0,
            "avg_latency": sum(latencies[rrf_k]) / len(latencies[rrf_k]) if latencies[rrf_k] else 0.0,
        }
    return metrics_by_rrf


def evaluate_ablation_shared(questions, sources, rag, generate_answers, progress_every=10):
    print("Starting ablation runs (shared retrieval)...")
    id_to_url = {s.get("id"): s.get("url") for s in sources}
    dense_cache = []
    sparse_cache = []

    for q in questions:
        dense_res, dense_idx = rag.dense_retrieval(q["question"], top_k=8)
        sparse_res, sparse_idx = rag.sparse_retrieval(q["question"], top_k=8)
        dense_cache.append((dense_res, dense_idx))
        sparse_cache.append((sparse_res, sparse_idx))

    runs = [
        {"mode": "dense", "top_k": 8, "top_n": 8, "rrf_k": 0},
        {"mode": "dense", "top_k": 5, "top_n": 5, "rrf_k": 0},
        {"mode": "sparse", "top_k": 8, "top_n": 8, "rrf_k": 0},
        {"mode": "sparse", "top_k": 5, "top_n": 5, "rrf_k": 0},
        {"mode": "hybrid", "top_k": 8, "top_n": 5, "rrf_k": 30},
        {"mode": "hybrid", "top_k": 8, "top_n": 5, "rrf_k": 60},
        {"mode": "hybrid", "top_k": 5, "top_n": 5, "rrf_k": 30},
        {"mode": "hybrid", "top_k": 5, "top_n": 5, "rrf_k": 60},
    ]

    ablation = []
    for run_idx, run in enumerate(runs, start=1):
        print(
            f"Ablation run {run_idx}/{len(runs)}: "
            f"{run['mode']} top_k={run['top_k']} top_n={run['top_n']} rrf_k={run['rrf_k']}"
        )
        mrr_sum = 0.0
        precision_at_5_sum = 0.0
        f1_sum = 0.0
        cp_sum = 0.0
        distinct_tokens = []
        latencies = []

        for idx, q in enumerate(questions, start=1):
            if progress_every and idx % progress_every == 0:
                print(f"  processed {idx}/{len(questions)} questions")
            dense_res, dense_idx = dense_cache[idx - 1]
            sparse_res, sparse_idx = sparse_cache[idx - 1]

            t0 = time.time()
            if run["mode"] == "dense":
                context = dense_res[:run["top_n"]]
            elif run["mode"] == "sparse":
                context = sparse_res[:run["top_n"]]
            else:
                rag.k = run["rrf_k"]
                dense_indices = dense_idx[:run["top_k"]]
                sparse_indices = sparse_idx[:run["top_k"]]
                context = rag.rrf_fusion(dense_indices, sparse_indices, top_n=run["top_n"])

            answer = ""
            if generate_answers:
                answer = rag.generate_answer(q["question"], context)
            latency = time.time() - t0
            latencies.append(latency)

            retrieved_urls = unique_preserve([c.get("url") for c in context if c.get("url")])
            correct_ids = q.get("source_ids", [])
            correct_urls = [id_to_url.get(sid) for sid in correct_ids if id_to_url.get(sid)]

            rank = None
            if correct_urls:
                for r_i, url in enumerate(retrieved_urls):
                    if url in correct_urls:
                        rank = r_i + 1
                        break
            reciprocal_rank = (1.0 / rank) if rank else 0.0
            mrr_sum += reciprocal_rank

            k = 5
            top_k_urls = retrieved_urls[:k]
            hits = sum(1 for url in top_k_urls if url in correct_urls) if correct_urls else 0
            precision_at_5_sum += hits / k

            f1 = token_f1(answer, q.get("ground_truth", "")) if generate_answers else 0.0
            f1_sum += f1

            cp = contextual_precision(answer, context) if generate_answers else 0.0
            cp_sum += cp

            distinct_tokens.extend(normalize_tokens(answer))

        total = len(questions) if questions else 1
        distinct_1 = (len(set(distinct_tokens)) / len(distinct_tokens)) if distinct_tokens else 0.0
        ablation.append({
            "mode": run["mode"],
            "top_k": run["top_k"],
            "top_n": run["top_n"],
            "rrf_k": run["rrf_k"],
            "mrr_url": mrr_sum / total,
            "precision_at_5_url": precision_at_5_sum / total,
            "answer_f1": f1_sum / total,
            "contextual_precision": cp_sum / total,
            "distinct_1": distinct_1,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        })
    print("ablation complete")
    return ablation


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_plot(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_architecture_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    boxes = [
        ("Docs", 0.05, 0.45),
        ("Chunking", 0.2, 0.45),
        ("BM25", 0.38, 0.65),
        ("Embeddings", 0.38, 0.25),
        ("FAISS", 0.55, 0.25),
        ("RRF", 0.7, 0.45),
        ("LLM", 0.85, 0.45),
    ]
    for label, x, y in boxes:
        rect = Rectangle((x, y), 0.12, 0.2, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.06, y + 0.1, label, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.2, 0.55), xytext=(0.17, 0.55), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.38, 0.75), xytext=(0.32, 0.55), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.38, 0.35), xytext=(0.32, 0.55), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.55, 0.35), xytext=(0.5, 0.35), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.7, 0.55), xytext=(0.62, 0.55), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.85, 0.55), xytext=(0.82, 0.55), arrowprops={"arrowstyle": "->"})
    save_plot(fig, path)


def plot_metric_comparison(metrics, path):
    labels = ["MRR", "P@5", "F1", "CtxPrec", "Distinct-1"]
    values = [
        metrics.get("mrr_url", 0.0),
        metrics.get("precision_at_5_url", 0.0),
        metrics.get("answer_f1", 0.0),
        metrics.get("contextual_precision", 0.0),
        metrics.get("distinct_1", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric Comparison")
    save_plot(fig, path)


def plot_score_distributions(results, path):
    f1 = [r.get("answer_f1", 0.0) for r in results]
    cp = [r.get("contextual_precision", 0.0) for r in results]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(f1, bins=10, color="#72B7B2", edgecolor="black")
    ax[0].set_title("Answer F1 Distribution")
    ax[1].hist(cp, bins=10, color="#F58518", edgecolor="black")
    ax[1].set_title("Contextual Precision Distribution")
    save_plot(fig, path)


def plot_retrieval_heatmap(error_analysis, path):
    categories = sorted(error_analysis.keys())
    failure_types = ["retrieval_failure", "context_failure", "generation_failure", "success"]
    matrix = []
    for cat in categories:
        row = [error_analysis.get(cat, {}).get(ft, 0) for ft in failure_types]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(failure_types)))
    ax.set_xticklabels(failure_types, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(failure_types)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=8)
    ax.set_title("Retrieval/Generation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_plot(fig, path)


def plot_response_times(results, path):
    times = [r.get("latency", 0.0) for r in results]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(times, bins=10, color="#E45756", edgecolor="black")
    ax.set_title("Response Time Distribution (s)")
    save_plot(fig, path)


def plot_ablation(ablation, path):
    if not ablation:
        return
    mode_scores = defaultdict(list)
    for row in ablation:
        mode_scores[row["mode"]].append(row.get("mrr_url", 0.0))
    modes = list(mode_scores.keys())
    avg_scores = [sum(mode_scores[m]) / len(mode_scores[m]) for m in modes]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(modes, avg_scores, color="#54A24B")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: Avg MRR by Mode")
    save_plot(fig, path)


def ensure_screenshots(output_dir):
    shots_dir = os.path.join(output_dir, "screenshots")
    ensure_dir(shots_dir)
    existing = sorted([f for f in os.listdir(shots_dir) if f.endswith(".png")])
    if len(existing) >= 3:
        return [os.path.join("screenshots", f) for f in existing[:3]]
    for i in range(len(existing) + 1, 4):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Screenshot Placeholder {i}", ha="center", va="center", fontsize=14)
        filename = f"screenshot_{i}.png"
        save_plot(fig, os.path.join(shots_dir, filename))
    existing = sorted([f for f in os.listdir(shots_dir) if f.endswith(".png")])
    return [os.path.join("screenshots", f) for f in existing[:3]]


def collect_error_examples(results, labels, max_per_type=3):
    examples = defaultdict(list)
    for r, label in zip(results, labels):
        if len(examples[label]) >= max_per_type:
            continue
        examples[label].append({
            "question_id": r.get("question_id"),
            "question": r.get("question"),
            "generated_answer": r.get("generated_answer"),
            "first_correct_url_rank": r.get("first_correct_url_rank"),
            "answer_f1": r.get("answer_f1"),
            "contextual_precision": r.get("contextual_precision"),
        })
    return {k: v for k, v in examples.items()}


def summarize_error_patterns(error_analysis):
    if not error_analysis:
        return []
    failure_counts = {}
    for cat, counts in error_analysis.items():
        failure_counts[cat] = sum(v for k, v in counts.items() if k != "success")
    top = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    patterns = []
    if top:
        patterns.append(f"Most failures by category: {top[0][0]} ({top[0][1]}).")
    return patterns


def write_html_report(path, summary):
    metrics = summary.get("metrics", {})
    adv = summary.get("adversarial", {})
    errors = summary.get("error_analysis", {})
    ablation = summary.get("ablation", [])
    plots = summary.get("plots", {})
    screenshots = summary.get("screenshots", [])
    details = summary.get("details", [])
    error_examples = summary.get("error_examples", {})

    html = []
    html.append("<html><head><meta charset='utf-8'><style>")
    html.append("body{font-family:Arial, sans-serif; margin:24px;} h1{margin-bottom:8px;} table{border-collapse:collapse; width:100%; margin:12px 0;} th,td{border:1px solid #ccc; padding:6px;} th{background:#f2f2f2;}")
    html.append(".section{margin-top:24px;} .small{color:#666; font-size:12px;} .img{max-width:100%; border:1px solid #ddd; padding:4px;}")
    html.append("</style></head><body>")
    html.append("<h1>RAG Evaluation Report</h1>")

    html.append("<div class='section'><h2>Overall Performance Summary</h2>")
    html.append("<p>MRR (URL), Answer F1, and Contextual Precision are the primary indicators of retrieval accuracy and grounded answer quality.</p>")
    html.append("<table><tr><th>Metric</th><th>Value</th></tr>")
    for k in ["mrr_url", "precision_at_5_url", "answer_f1", "contextual_precision", "distinct_1", "avg_latency"]:
        html.append(f"<tr><td>{k}</td><td>{metrics.get(k, 0.0):.4f}</td></tr>")
    html.append("</table></div>")

    html.append("<div class='section'><h2>Custom Metrics Justification</h2>")
    html.append("<h3>Answer F1</h3>")
    html.append("<p><b>Why chosen:</b> Measures overlap between generated answers and ground truth while allowing partial credit. Useful for open-ended answers.</p>")
    html.append("<p><b>Calculation:</b> precision = |overlap|/|pred|, recall = |overlap|/|gt|, F1 = 2·precision·recall/(precision+recall).</p>")
    html.append("<p><b>Interpretation:</b> Higher is better; 1.0 is perfect lexical match.</p>")
    html.append("<h3>Contextual Precision</h3>")
    html.append("<p><b>Why chosen:</b> Measures grounding by checking how much of the answer appears in retrieved context, indicating faithfulness.</p>")
    html.append("<p><b>Calculation:</b> CP = |answer tokens ∩ context tokens| / |answer tokens|.</p>")
    html.append("<p><b>Interpretation:</b> Higher is better; low values suggest hallucination risk.</p>")
    html.append("</div>")

    if plots.get("architecture"):
        html.append("<div class='section'><h2>Architecture Diagram</h2>")
        html.append(f"<img class='img' src='{plots['architecture']}'/>")
        html.append("</div>")

    html.append("<div class='section'><h2>Visualizations</h2>")
    for key in ["metric_comparison", "score_distributions", "retrieval_heatmap", "response_times", "ablation_plot"]:
        if plots.get(key):
            html.append(f"<img class='img' src='{plots[key]}'/>")
    html.append("</div>")

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
    patterns = summary.get("error_patterns", [])
    if patterns:
        html.append("<div class='small'><b>Patterns:</b><ul>")
        for p in patterns:
            html.append(f"<li>{p}</li>")
        html.append("</ul></div>")

    if error_examples:
        html.append("<div class='section'><h3>Failure Examples</h3>")
        for label, examples in error_examples.items():
            html.append(f"<h4>{label}</h4><table>")
            html.append("<tr><th>ID</th><th>Question</th><th>Answer</th><th>Rank</th><th>F1</th><th>CtxPrec</th></tr>")
            for ex in examples:
                html.append(
                    "<tr>"
                    f"<td>{ex.get('question_id')}</td>"
                    f"<td>{ex.get('question')}</td>"
                    f"<td>{ex.get('generated_answer')}</td>"
                    f"<td>{ex.get('first_correct_url_rank')}</td>"
                    f"<td>{ex.get('answer_f1'):.4f}</td>"
                    f"<td>{ex.get('contextual_precision'):.4f}</td>"
                    "</tr>"
                )
            html.append("</table>")
        html.append("</div>")

    if ablation:
        html.append("<div class='section'><h2>Ablation Grid (Summary)</h2><table>")
        html.append("<tr><th>mode</th><th>top_k</th><th>top_n</th><th>rrf_k</th><th>mrr_url</th><th>precision_at_5_url</th></tr>")
        for row in ablation:
            html.append(
                "<tr>"
                f"<td>{row['mode']}</td>"
                f"<td>{row['top_k']}</td>"
                f"<td>{row['top_n']}</td>"
                f"<td>{row['rrf_k']}</td>"
                f"<td>{row['mrr_url']:.4f}</td>"
                f"<td>{row['precision_at_5_url']:.4f}</td>"
                "</tr>"
            )
        html.append("</table></div>")

    html.append("<div class='section'><h2>Results Table</h2><table>")
    html.append("<tr><th>ID</th><th>Question</th><th>Ground Truth</th><th>Generated Answer</th><th>MRR (1/rank)</th><th>F1</th><th>CtxPrec</th><th>Time (s)</th></tr>")
    for row in details:
        html.append(
            "<tr>"
            f"<td>{row.get('question_id')}</td>"
            f"<td>{row.get('question')}</td>"
            f"<td>{row.get('ground_truth')}</td>"
            f"<td>{row.get('generated_answer')}</td>"
            f"<td>{row.get('reciprocal_rank'):.4f}</td>"
            f"<td>{row.get('answer_f1'):.4f}</td>"
            f"<td>{row.get('contextual_precision'):.4f}</td>"
            f"<td>{row.get('latency'):.4f}</td>"
            "</tr>"
        )
    html.append("</table></div>")

    html.append("<div class='section'><h2>System Screenshots</h2>")
    if screenshots:
        for shot in screenshots:
            html.append(f"<img class='img' src='{shot}'/>")
    else:
        html.append("<p>No screenshots found.</p>")
    html.append("</div>")

    html.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(html))


def write_pdf_report(path, summary):
    if canvas is None or letter is None:
        raise RuntimeError("reportlab is not installed. Install it to generate PDF reports.")

    def draw_image(c, img_path, y, max_width, max_height):
        if ImageReader is None or not os.path.exists(img_path):
            return y
        img = ImageReader(img_path)
        iw, ih = img.getSize()
        scale = min(max_width / iw, max_height / ih)
        w = iw * scale
        h = ih * scale
        c.drawImage(img, 48, y - h, width=w, height=h)
        return y - h - 12

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
    line("Overall Performance Summary", bold=True)
    for k in ["mrr_url", "precision_at_5_url", "answer_f1", "contextual_precision", "distinct_1", "avg_latency"]:
        line(f"{k}: {summary.get('metrics', {}).get(k, 0.0):.4f}")
    line("")
    line("Custom Metrics Justification", bold=True)
    line("Answer F1: precision/recall overlap; F1 = 2PR/(P+R).")
    line("Contextual Precision: |answer tokens ∩ context tokens| / |answer tokens|.")
    line("")

    plots = summary.get("plots", {})
    if plots.get("architecture"):
        line("Architecture Diagram", bold=True)
        y = draw_image(c, os.path.join(summary["output_dir"], plots["architecture"]), y, width - 96, 160)

    for key in ["metric_comparison", "score_distributions", "retrieval_heatmap", "response_times", "ablation_plot"]:
        if plots.get(key):
            line(key.replace("_", " ").title(), bold=True)
            y = draw_image(c, os.path.join(summary["output_dir"], plots[key]), y, width - 96, 160)

    line("Adversarial Testing", bold=True)
    adv = summary.get("adversarial", {})
    line(f"paraphrase_hit_rate: {adv.get('paraphrase_hit_rate', 0.0):.4f}")
    line(f"unanswerable_hallucination_rate: {adv.get('unanswerable_hallucination_rate', 0.0):.4f}")
    line("")

    line("Error Analysis (by category)", bold=True)
    for cat, counts in summary.get("error_analysis", {}).items():
        line(f"{cat}: {counts}")
    line("")
    for p in summary.get("error_patterns", []):
        line(f"Pattern: {p}")
    line("")

    line("Results Table (first 10, includes MRR contribution)", bold=True)
    for row in summary.get("details", [])[:10]:
        line(
            f"{row.get('question_id')}: MRR={row.get('reciprocal_rank'):.2f} "
            f"F1={row.get('answer_f1'):.2f} "
            f"Ctx={row.get('contextual_precision'):.2f} "
            f"T={row.get('latency'):.2f}s"
        )

    screenshots = summary.get("screenshots", [])
    if screenshots:
        line("Screenshots", bold=True)
        for shot in screenshots:
            y = draw_image(c, os.path.join(summary["output_dir"], shot), y, width - 96, 160)

    c.save()


def generate_plots(summary, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    ensure_dir(plot_dir)
    plots = {}

    arch_path = os.path.join(plot_dir, "architecture.png")
    generate_architecture_diagram(arch_path)
    plots["architecture"] = os.path.join("plots", "architecture.png")

    metric_path = os.path.join(plot_dir, "metric_comparison.png")
    plot_metric_comparison(summary.get("metrics", {}), metric_path)
    plots["metric_comparison"] = os.path.join("plots", "metric_comparison.png")

    dist_path = os.path.join(plot_dir, "score_distributions.png")
    plot_score_distributions(summary.get("details", []), dist_path)
    plots["score_distributions"] = os.path.join("plots", "score_distributions.png")

    heatmap_path = os.path.join(plot_dir, "retrieval_heatmap.png")
    plot_retrieval_heatmap(summary.get("error_analysis", {}), heatmap_path)
    plots["retrieval_heatmap"] = os.path.join("plots", "retrieval_heatmap.png")

    time_path = os.path.join(plot_dir, "response_times.png")
    plot_response_times(summary.get("details", []), time_path)
    plots["response_times"] = os.path.join("plots", "response_times.png")

    if summary.get("ablation"):
        ablation_path = os.path.join(plot_dir, "ablation_plot.png")
        plot_ablation(summary.get("ablation", []), ablation_path)
        plots["ablation_plot"] = os.path.join("plots", "ablation_plot.png")

    return plots


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
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation grid.")
    parser.add_argument("--ablation-generate", action="store_true", help="Generate answers during ablation (slow).")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    print("Pipeline started.")

    print("Loading questions...")
    qa_data = load_questions(args.questions)
    questions = qa_data.get("questions", [])
    sources = qa_data.get("sources", [])
    if not questions:
        raise SystemExit("No questions found.")

    print("Loading data and building index...")
    data_loader = DataLoader()
    documents = data_loader.load_all_data()
    indexer = Indexer()
    indexer.build_index(documents)
    rag = RAGEngine(indexer)

    print("Running main evaluation...")
    metrics, results = evaluate_questions(
        questions, sources, rag, args.mode, args.top_k, args.top_n, args.rrf_k, not args.no_generate
    )

    print("Running adversarial tests...")
    sample_n = min(args.adversarial_sample, len(questions))
    sample = random.sample(questions, sample_n)
    adversarial = run_adversarial(
        sample, sources, rag, args.mode, args.top_k, args.top_n, args.rrf_k, not args.no_generate
    )

    print("Computing error analysis...")
    labels = build_error_labels(results)
    by_cat = defaultdict(Counter)
    for r, label in zip(results, labels):
        by_cat[r.get("category", "unknown")][label] += 1
    error_analysis = {k: dict(v) for k, v in by_cat.items()}
    error_examples = collect_error_examples(results, labels)
    error_patterns = summarize_error_patterns(error_analysis)

    ablation = []
    if not args.skip_ablation:
        print("Running ablation grid (shared retrieval, 8 runs)...")
        print("Ablation runs: dense(8/8), dense(5/5), sparse(8/8), sparse(5/5), "
              "hybrid(8/5, rrf=30), hybrid(8/5, rrf=60), hybrid(5/5, rrf=30), hybrid(5/5, rrf=60)")
        if not args.ablation_generate:
            print("Ablation note: answer generation disabled for speed.")
        ablation = evaluate_ablation_shared(
            questions, sources, rag, generate_answers=(args.ablation_generate and not args.no_generate)
        )

    print("Generating plots and screenshots...")
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
        "error_examples": error_examples,
        "error_patterns": error_patterns,
        "ablation": ablation,
        "details": results,
        "output_dir": args.output_dir,
    }

    summary["plots"] = generate_plots(summary, args.output_dir)
    summary["screenshots"] = ensure_screenshots(args.output_dir)

    print("Writing JSON/CSV/HTML/PDF reports...")
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
