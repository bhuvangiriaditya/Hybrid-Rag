import json
import random
import re
import os


def clean_text(text):
    text = text.replace("This is an accepted version of this page ", "")
    text = re.sub(r"\\[[0-9]+\\]", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def first_sentence(text):
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\\s+", text)
    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            return p
    return text[:200].strip()


def shorten(phrase, max_words=14):
    words = phrase.split()
    return " ".join(words[:max_words])


def load_articles(corpus_path):
    with open(corpus_path, "r") as f:
        data = json.load(f)
    # If article-level with chunks
    if data and isinstance(data, list) and "chunks" in data[0]:
        articles = []
        for d in data:
            if d.get("text"):
                full_text = d.get("text")
            else:
                # Reconstruct from chunks
                chunks = sorted(d.get("chunks", []), key=lambda c: c.get("chunk_index", 0))
                full_text = " ".join([c.get("text", "") for c in chunks])
            articles.append({
                "title": d.get("title"),
                "url": d.get("url"),
                "text": full_text,
                "chunks": d.get("chunks", [])
            })
        return articles
    return data


def build_questions(corpus_path="data/scraped_all.json", out_path="data/rag_questions.json"):
    if not os.path.exists(corpus_path):
        corpus_path = "data/scraped_fixed.json"
    docs = load_articles(corpus_path)

    sources = []
    for i, d in enumerate(docs, start=1):
        sources.append({
            "id": f"S{i:03d}",
            "title": d["title"],
            "url": d["url"],
        })
    id_by_title = {s["title"]: s["id"] for s in sources}

    def get_relevant_chunk(doc):
        chunks = doc.get("chunks", [])
        if chunks:
            chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
            first = chunks[0]
            return first.get("text", ""), first.get("chunk_id")
        # Fallback to short snippet if chunks missing
        return first_sentence(doc.get("text", "")), None

    random.seed(42)
    random.shuffle(docs)

    questions = []

    # 1) Factual (50)
    for d in docs:
        if len(questions) >= 50:
            break
        title = d["title"]
        q = f"What is {title}?"
        expected = first_sentence(d["text"])
        chunk_text, chunk_id = get_relevant_chunk(d)
        questions.append({
            "question": q,
            "ground_truth": chunk_text,
            "expected_answer": expected,
            "ground_truth_chunk_ids": [chunk_id] if chunk_id else [],
            "source_ids": [id_by_title[title]],
            "category": "factual",
        })

    # 2) Comparative (20)
    comp_pairs = []
    for i in range(0, len(docs) - 1, 2):
        if len(comp_pairs) >= 20:
            break
        d1, d2 = docs[i], docs[i + 1]
        if d1["title"] == d2["title"]:
            continue
        comp_pairs.append((d1, d2))

    for d1, d2 in comp_pairs:
        q = (
            f"Compare {d1['title']} and {d2['title']} in one sentence each: "
            "what does each describe or study?"
        )
        a1 = first_sentence(d1["text"])
        a2 = first_sentence(d2["text"])
        expected = f"{d1['title']}: {a1} {d2['title']}: {a2}"
        c1, c1_id = get_relevant_chunk(d1)
        c2, c2_id = get_relevant_chunk(d2)
        chunk_text = f"{d1['title']}: {c1} {d2['title']}: {c2}"
        questions.append({
            "question": q,
            "ground_truth": chunk_text,
            "expected_answer": expected,
            "ground_truth_chunk_ids": [c1_id, c2_id],
            "source_ids": [id_by_title[d1["title"]], id_by_title[d2["title"]]],
            "category": "comparative",
        })

    # 3) Inferential (15)
    patterns = [
        (re.compile(r"is the scientific study of ([^.]+)\\.", re.I),
         "Which field is the scientific study of {obj}?"),
        (re.compile(r"is the study of ([^.]+)\\.", re.I),
         "Which field is the study of {obj}?"),
        (re.compile(r"is a branch of ([^.]+)\\.", re.I),
         "Which field is described as a branch of {obj}?"),
        (re.compile(r"is a field of study that ([^.]+)\\.", re.I),
         "Which field is described as a field of study that {obj}?"),
    ]

    inferential_candidates = []
    for d in docs:
        text = clean_text(d["text"])
        for pat, template in patterns:
            m = pat.search(text)
            if m:
                obj = shorten(m.group(1).strip())
                q = template.format(obj=obj)
                inferential_candidates.append((q, d))
                break

    seen_q = set()
    inferential_added = 0
    for q, d in inferential_candidates:
        if inferential_added >= 15:
            break
        if q in seen_q:
            continue
        seen_q.add(q)
        chunk_text, chunk_id = get_relevant_chunk(d)
        questions.append({
            "question": q,
            "ground_truth": chunk_text,
            "expected_answer": d["title"],
            "ground_truth_chunk_ids": [chunk_id] if chunk_id else [],
            "source_ids": [id_by_title[d["title"]]],
            "category": "inferential",
        })
        inferential_added += 1

    while inferential_added < 15:
        d = random.choice(docs)
        q = f"Based on its description, which field or concept is being defined here: '{shorten(first_sentence(d['text']), 12)}'?"
        if q in seen_q:
            continue
        seen_q.add(q)
        chunk_text, chunk_id = get_relevant_chunk(d)
        questions.append({
            "question": q,
            "ground_truth": chunk_text,
            "expected_answer": d["title"],
            "ground_truth_chunk_ids": [chunk_id] if chunk_id else [],
            "source_ids": [id_by_title[d["title"]]],
            "category": "inferential",
        })
        inferential_added += 1

    # 4) Multi-hop (15)
    mentions = []
    for d in docs:
        text_low = clean_text(d["text"]).lower()
        for other_title in id_by_title.keys():
            if other_title == d["title"]:
                continue
            if len(other_title) < 6:
                continue
            t_low = other_title.lower()
            if re.search(r"\\b" + re.escape(t_low) + r"\\b", text_low):
                mentions.append((d, other_title))

    multi_added = 0
    used_pairs = set()
    for d1, title2 in mentions:
        if multi_added >= 15:
            break
        pair_key = tuple(sorted([d1["title"], title2]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        d2 = next(doc for doc in docs if doc["title"] == title2)
        q = (
            f"The {d1['title']} article mentions {d2['title']}. "
            f"Using both sources, state what {d1['title']} is and what {d2['title']} is."
        )
        a1 = first_sentence(d1["text"])
        a2 = first_sentence(d2["text"])
        expected = f"{d1['title']}: {a1} {d2['title']}: {a2}"
        c1, c1_id = get_relevant_chunk(d1)
        c2, c2_id = get_relevant_chunk(d2)
        chunk_text = f"{d1['title']}: {c1} {d2['title']}: {c2}"
        questions.append({
            "question": q,
            "ground_truth": chunk_text,
            "expected_answer": expected,
            "ground_truth_chunk_ids": [c1_id, c2_id],
            "source_ids": [id_by_title[d1["title"]], id_by_title[d2["title"]]],
            "category": "multi-hop",
        })
        multi_added += 1

    # Ensure exact 100 questions
    if len(questions) > 100:
        questions = questions[:100]
    elif len(questions) < 100:
        needed = 100 - len(questions)
        for d in docs:
            if needed == 0:
                break
            title = d["title"]
            q = f"What is {title}?"
            expected = first_sentence(d["text"])
            chunk_text, chunk_id = get_relevant_chunk(d)
            questions.append({
                "question": q,
                "ground_truth": chunk_text,
                "expected_answer": expected,
                "ground_truth_chunk_ids": [chunk_id] if chunk_id else [],
                "source_ids": [id_by_title[title]],
                "category": "factual",
            })
            needed -= 1

    for i, q in enumerate(questions, start=1):
        q["id"] = i

    output = {
        "description": "Evaluation questions with ground-truth answers derived from the local Wikipedia-based corpus (scraped_fixed.json).",
        "total_questions": len(questions),
        "sources": sources,
        "questions": questions,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = out_path.replace(".json", ".csv")
    with open(csv_path, "w") as f:
        f.write("id,category,question,ground_truth,expected_answer,ground_truth_chunk_ids,source_ids\n")
        for q in questions:
            source_ids = "|".join(q.get("source_ids", []))
            chunk_ids = "|".join(q.get("ground_truth_chunk_ids", []))
            row = [
                str(q.get("id", "")),
                q.get("category", "").replace(",", " "),
                q.get("question", "").replace(",", " "),
                q.get("ground_truth", "").replace(",", " "),
                q.get("expected_answer", "").replace(",", " "),
                chunk_ids.replace(",", " "),
                source_ids.replace(",", " "),
            ]
            f.write(",".join(row) + "\n")

    print(f"Wrote {len(questions)} questions to {out_path}")
    print(f"Wrote {len(questions)} questions to {csv_path}")


if __name__ == "__main__":
    build_questions()
