import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import streamlit as st

from data_loader import DataLoader
from indexer import Indexer
from rag_engine import RAGEngine


def load_questions(filepath="data/rag_questions.json"):
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


def extract_entities(text):
    if not text:
        return []
    return re.findall(r"\\b[A-Z][a-zA-Z0-9-]{2,}\\b", text)


def entity_coverage(answer, ground_truth):
    entities = extract_entities(ground_truth)
    if not entities:
        return 0.0
    answer_lower = answer.lower() if answer else ""
    covered = sum(1 for e in entities if e.lower() in answer_lower)
    return covered / len(entities)


def unique_preserve(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def ensure_indexer():
    if "indexer" in st.session_state:
        indexer = st.session_state["indexer"]
        if indexer.faiss_index is None:
            with st.spinner("Index not built. Building now..."):
                data_loader = DataLoader()
                documents = data_loader.load_all_data()
                indexer.build_index(documents)
        return indexer
    with st.spinner("Loading data and building index..."):
        data_loader = DataLoader()
        documents = data_loader.load_all_data()
        indexer = Indexer()
        indexer.build_index(documents)
        st.session_state["indexer"] = indexer
        st.session_state["documents"] = documents
    return st.session_state["indexer"]


def ensure_rag():
    if "rag_engine" in st.session_state:
        return st.session_state["rag_engine"]
    indexer = ensure_indexer()
    with st.spinner("Loading LLM (google/flan-t5-base)..."):
        rag = RAGEngine(indexer)
        st.session_state["rag_engine"] = rag
    return st.session_state["rag_engine"]


def llm_generate(rag, prompt, max_length=256):
    inputs = rag.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = rag.model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return rag.tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_json_scores(text):
    if not text:
        return {}
    # Extract JSON object if the model wraps it in text.
    match = re.search(r"\\{.*\\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return {}
    scores = obj.get("scores", obj)
    out = {}
    for key in ("factuality", "completeness", "relevance", "coherence"):
        if key in scores:
            try:
                out[key] = int(scores[key])
            except Exception:
                pass
    return out


def heuristic_scores(answer, context_chunks, ground_truth):
    f1 = token_f1(answer, ground_truth)
    cp = contextual_precision(answer, context_chunks)
    answer_tokens = normalize_tokens(answer)
    length_score = min(len(answer_tokens) / 20.0, 1.0)

    def to_5(x):
        return max(1, min(5, int(round(1 + 4 * x))))

    return {
        "factuality": to_5(cp),
        "completeness": to_5(f1),
        "relevance": to_5(cp),
        "coherence": to_5(length_score),
    }

def parse_judge_scores(text):
    if not text:
        return {}
    patterns = {
        "factuality": r"Factuality\\s*[:=]\\s*(\\d)",
        "completeness": r"Completeness\\s*[:=]\\s*(\\d)",
        "relevance": r"Relevance\\s*[:=]\\s*(\\d)",
        "coherence": r"Coherence\\s*[:=]\\s*(\\d)",
    }
    scores = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            scores[key] = int(m.group(1))
    if len(scores) == 4:
        return scores

    # Fallback: extract the first 4 integers in [1-5] order
    nums = re.findall(r"\\b([1-5])\\b", text)
    if len(nums) >= 4:
        scores = {
            "factuality": int(nums[0]),
            "completeness": int(nums[1]),
            "relevance": int(nums[2]),
            "coherence": int(nums[3]),
        }
    return scores


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
    rag = ensure_rag()
    id_to_url = {s.get("id"): s.get("url") for s in sources}
    results = []
    mrr_sum = 0.0
    precision_at_5_sum = 0.0
    f1_sum = 0.0
    cp_sum = 0.0
    distinct_1_tokens = []
    latencies = []

    progress = st.progress(0)
    for i, q in enumerate(questions):
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

        expected_answer = q.get("expected_answer", q.get("ground_truth", ""))
        f1 = token_f1(answer, expected_answer) if generate_answers else 0.0
        f1_sum += f1

        cp = contextual_precision(answer, context) if generate_answers else 0.0
        cp_sum += cp

        distinct_1_tokens.extend(normalize_tokens(answer))

        results.append({
            "question_id": q.get("id"),
            "question": q.get("question"),
            "category": q.get("category"),
            "ground_truth": q.get("ground_truth"),
            "expected_answer": expected_answer,
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

        progress.progress((i + 1) / len(questions))
    progress.empty()

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


def compute_confidence(rank, contextual_prec):
    rank_score = (1.0 / rank) if rank else 0.0
    return 0.5 * rank_score + 0.5 * contextual_prec


def calibration_bins(results, bins=10, f1_threshold=0.2):
    bucket = {i: {"conf": [], "acc": []} for i in range(bins)}
    for r in results:
        conf = compute_confidence(
            r.get("first_correct_url_rank"), r.get("contextual_precision", 0.0)
        )
        acc = 1.0 if r.get("answer_f1", 0.0) >= f1_threshold else 0.0
        idx = min(int(conf * bins), bins - 1)
        bucket[idx]["conf"].append(conf)
        bucket[idx]["acc"].append(acc)
    x = []
    y = []
    for i in range(bins):
        if not bucket[i]["conf"]:
            continue
        x.append(sum(bucket[i]["conf"]) / len(bucket[i]["conf"]))
        y.append(sum(bucket[i]["acc"]) / len(bucket[i]["acc"]))
    return x, y


def render_eval_dashboard():
    st.title("RAG Evaluation Dashboard")

    qa_data = load_questions()
    questions = qa_data.get("questions", [])
    sources = qa_data.get("sources", [])

    if not questions:
        st.error("No questions found in data/rag_questions.json.")
        st.stop()

    with st.sidebar:
        st.header("Evaluation Controls")
        mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"])
        top_k = st.slider("Top-K Retrieval", 1, 20, 5)
        top_n = st.slider("Context Chunks (N)", 1, 10, 5)
        rrf_k = st.slider("RRF k", 1, 100, 60)
        generate_answers = st.checkbox("Generate Answers (slow)", value=True)
        run_eval = st.button("Run Evaluation")
        load_prev = st.button("Load evaluation_results.json")

    tabs = st.tabs([
        "Overview",
        "Explorer",
        "Adversarial Testing",
        "Ablation Study",
        "Error Analysis",
        # "LLM-as-Judge",
        "Calibration"
    ])

    if load_prev and os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r") as f:
            prev = json.load(f)
        st.session_state["eval_metrics"] = prev.get("metrics", {})
        st.session_state["eval_results"] = prev.get("details", [])

    if run_eval:
        metrics, results = evaluate_questions(
            questions, sources, mode, top_k, top_n, rrf_k, generate_answers
        )
        st.session_state["eval_metrics"] = metrics
        st.session_state["eval_results"] = results

    metrics = st.session_state.get("eval_metrics", {})
    results = st.session_state.get("eval_results", [])

    with tabs[0]:
        st.subheader("Dataset Overview")
        cat_counts = Counter(q.get("category", "unknown") for q in questions)
        st.write({"total_questions": len(questions), "categories": dict(cat_counts)})

        st.subheader("Core Metrics")
        if metrics:
            st.metric("MRR (URL)", f"{metrics.get('mrr_url', 0.0):.2f}")
            st.caption("Average of 1/rank for the first correct URL. Higher means correct sources appear earlier.")

            st.metric("Precision@5 (URL)", f"{metrics.get('precision_at_5_url', 0.0):.2f}")
            st.caption("Fraction of correct URLs in the top-5 retrieved URLs. Higher means cleaner top-5.")

            if generate_answers:
                st.metric("Answer F1", f"{metrics.get('answer_f1', 0.0):.2f}")
                st.caption("Token overlap F1 between generated answer and ground truth (partial credit).")

                st.metric("Contextual Precision", f"{metrics.get('contextual_precision', 0.0):.2f}")
                st.caption("Share of answer tokens found in retrieved context. Higher means more grounded answers.")

                st.metric("Distinct-1", f"{metrics.get('distinct_1', 0.0):.2f}")
                st.caption("Unique tokens / total tokens in answers. Higher means less repetition.")

            st.metric("Avg Latency (s)", f"{metrics.get('avg_latency', 0.0):.2f}")
            st.caption("Average end-to-end time per question (retrieval + generation).")
        else:
            st.info("Run evaluation to compute metrics.")

    with tabs[1]:
        st.subheader("Interactive Explorer")
        if results:
            q_ids = [r.get("question_id") for r in results]
            selected = st.selectbox("Question ID", q_ids)
            record = next(r for r in results if r.get("question_id") == selected)
            st.write(record)
        else:
            st.info("Run evaluation to explore results.")

    with tabs[2]:
        st.subheader("Adversarial Testing")
        sample_n = st.slider("Sample size", 5, min(50, len(questions)), 20)
        sample = random.sample(questions, sample_n)
        rag = ensure_rag()

        paraphrase_hits = 0
        unanswerable_hallucinations = 0
        paraphrase_details = []
        unanswerable_details = []
        id_to_url = {s.get("id"): s.get("url") for s in sources}

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

        st.write({
            "paraphrase_hit_rate": paraphrase_hits / sample_n if sample_n else 0.0,
            "unanswerable_hallucination_rate": unanswerable_hallucinations / sample_n if sample_n else 0.0,
        })
        st.caption(
            "Paraphrase hit rate: for each sampled question, we paraphrase it and check whether the top "
            "retrieved URL is still one of the correct source URLs. "
            "Unanswerable hallucination rate: for each sampled question, we craft an unanswerable variant "
            "and count cases where the model still produces a non-empty answer with low contextual precision (< 0.2)."
        )
        st.subheader("Paraphrase Details (Sample)")
        st.json(paraphrase_details)
        st.subheader("Unanswerable Details (Sample)")
        st.json(unanswerable_details)

    with tabs[3]:
        st.subheader("Ablation Study")
        st.write(
            "This grid runs controlled comparisons across retrieval modes and parameters to show which "
            "setup performs best. For each combination, it runs retrieval (and optionally generation) "
            "and records the metrics below. Use it to understand whether dense-only, sparse-only, or "
            "hybrid retrieval is most effective for this dataset, and how sensitive performance is to "
            "Top-K, Top-N, and RRF k."
        )
        st.caption(
            "Grid settings: Dense/Sparse use Top-K = {3, 5, 8} with Top-N=None (not used). "
            "Hybrid uses Top-K = {3, 5, 8}, Top-N = {3, 5}, RRF k = {30, 60}. "
            "When 'Include answer generation' is enabled, the LLM is run for every combination, so "
            "this can be slow."
        )
        ablate_generate = st.checkbox("Include answer generation in ablation (slow)", value=False)
        if st.button("Run Ablation Grid"):
            rows = []
            # Dense + Sparse (Top-N not used, so display None; evaluate with top_n=top_k)
            for m in ["dense", "sparse"]:
                for k in [3, 5, 8]:
                    met, _ = evaluate_questions(
                        questions, sources, m, k, k, 60, ablate_generate
                    )
                    rows.append({
                        "mode": m,
                        "top_k": k,
                        "top_n": None,
                        "rrf_k": None,
                        "mrr_url": met.get("mrr_url", 0.0),
                        "precision_at_5_url": met.get("precision_at_5_url", 0.0),
                        "answer_f1": met.get("answer_f1", 0.0),
                        "contextual_precision": met.get("contextual_precision", 0.0),
                    })

            # Hybrid (Top-N + RRF k apply)
            for k in [3, 5, 8]:
                for n in [3, 5]:
                    for rk in [30, 60]:
                        met, _ = evaluate_questions(
                            questions, sources, "hybrid", k, n, rk, ablate_generate
                        )
                        rows.append({
                            "mode": "hybrid",
                            "top_k": k,
                            "top_n": n,
                            "rrf_k": rk,
                            "mrr_url": met.get("mrr_url", 0.0),
                            "precision_at_5_url": met.get("precision_at_5_url", 0.0),
                            "answer_f1": met.get("answer_f1", 0.0),
                            "contextual_precision": met.get("contextual_precision", 0.0),
                        })
            st.session_state["ablation_rows"] = rows

        if "ablation_rows" in st.session_state:
            st.dataframe(st.session_state["ablation_rows"])

    with tabs[4]:
        st.subheader("Error Analysis")
        if results:
            st.caption(
                "Failure labels: retrieval_failure = no correct URL found; "
                "context_failure = low contextual precision; "
                "generation_failure = correct URL found but low Answer F1; "
                "success = correct URL found and answer quality above threshold."
            )
            labels = build_error_labels(results)
            by_cat = defaultdict(Counter)
            for r, label in zip(results, labels):
                by_cat[r.get("category", "unknown")][label] += 1
            st.write({k: dict(v) for k, v in by_cat.items()})
        else:
            st.info("Run evaluation to see error breakdowns.")

    # with tabs[5]:
    #     st.subheader("LLM-as-Judge (Flan-T5)")
    #     st.write("Uses the local Flan-T5 model to score factuality, completeness, relevance, and coherence.")
    #     judge_n = st.slider("Questions to judge", 1, min(20, len(questions)), 5, key="judge_n")
    #     if st.button("Run LLM-as-Judge"):
    #         rag = ensure_rag()
    #         judged = []
    #         fallback_count = 0
    #         for q in questions[:judge_n]:
    #             context, _, _ = retrieve_context(rag, q["question"], mode, top_k, top_n, rrf_k)
    #             answer = rag.generate_answer(q["question"], context)
    #             context_text = " ".join([c.get("text", "") for c in context])
    #             prompt = (
    #                 "You are a strict evaluator. Use ONLY the context to judge the answer. "
    #                 "Return a single JSON object with this exact schema and integer scores 1-5:\n"
    #                 "{\n"
    #                 "  \"scores\": {\"factuality\": 1, \"completeness\": 1, \"relevance\": 1, \"coherence\": 1},\n"
    #                 "  \"explanation\": \"one short sentence\"\n"
    #                 "}\n"
    #                 "No extra text.\n"
    #                 f"Context: {context_text}\n"
    #                 f"Question: {q['question']}\n"
    #                 f"Answer: {answer}\n"
    #             )
    #             verdict = llm_generate(rag, prompt, max_length=256)
    #             score_map = parse_json_scores(verdict) or parse_judge_scores(verdict)
    #             used_fallback = False
    #             if not score_map:
    #                 score_map = heuristic_scores(answer, context, q.get("ground_truth", ""))
    #                 used_fallback = True
    #                 fallback_count += 1
    #             verdict_clean = (
    #                 f"F:{score_map.get('factuality','-')} "
    #                 f"C:{score_map.get('completeness','-')} "
    #                 f"R:{score_map.get('relevance','-')} "
    #                 f"Co:{score_map.get('coherence','-')}"
    #             )
    #             judged.append({
    #                 "question": q["question"],
    #                 "answer": answer,
    #                 "verdict": verdict_clean,
    #                 "verdict_raw": verdict,
    #                 "scores": score_map,
    #                 "fallback_used": used_fallback,
    #             })
    #         st.session_state["llm_judge"] = judged
    #         st.session_state["llm_judge_fallbacks"] = fallback_count

    #     if "llm_judge" in st.session_state:
    #         st.dataframe(st.session_state["llm_judge"])
    #         if st.session_state.get("llm_judge_fallbacks", 0) > 0:
    #             st.warning(
    #                 f"Fallback scoring used for {st.session_state['llm_judge_fallbacks']} "
    #                 "rows because the LLM output was not parseable."
    #             )

    with tabs[5]:
        st.subheader("Confidence Calibration")
        if results:
            x, y = calibration_bins(results)
            chart_data = [{"confidence": x[i], "accuracy": y[i]} for i in range(len(x))]
            st.vega_lite_chart(
                chart_data,
                {
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": "confidence", "type": "quantitative"},
                        "y": {"field": "accuracy", "type": "quantitative"},
                    },
                },
            )
            st.write("Confidence blends URL rank and contextual precision. Accuracy is F1 >= 0.2.")
        else:
            st.info("Run evaluation to see calibration curves.")


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
    render_eval_dashboard()
