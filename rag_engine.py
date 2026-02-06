import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class RAGEngine:
    def __init__(self, indexer):
        self.indexer = indexer
        # Using Flan-T5-small for speed/local feasibility, or base if resources allow
        self.model_name = "google/flan-t5-base" 
        print(f"Loading LLM: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.k = 60 # RRF constant

    def dense_retrieval(self, query, top_k=5):
        query_vector = self.indexer.embedding_model.encode([query])
        distances, indices = self.indexer.faiss_index.search(np.array(query_vector, dtype=np.float32), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append(self.indexer.chunks[idx])
        return results, indices[0]

    def sparse_retrieval(self, query, top_k=5):
        tokenized_query = query.lower().split()
        # get_top_n returns text, but we need indices to match with chunks.
        # BM25 implementation in rank_bm25 doesn't give indices easily with get_top_n
        # So we get all scores and sort.
        scores = self.indexer.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [self.indexer.chunks[i] for i in top_indices]
        return results, top_indices

    def rrf_fusion(self, dense_indices, sparse_indices, top_n=5):
        """
        Reciprocal Rank Fusion
        score = 1 / (k + rank)
        """
        rrf_scores = {}

        # Dense Ranks
        for rank, idx in enumerate(dense_indices):
            if idx == -1: continue
            if idx not in rrf_scores: rrf_scores[idx] = 0
            rrf_scores[idx] += 1 / (self.k + rank + 1)

        # Sparse Ranks
        for rank, idx in enumerate(sparse_indices):
            if idx not in rrf_scores: rrf_scores[idx] = 0
            rrf_scores[idx] += 1 / (self.k + rank + 1)

        # Sort by score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_sorted = sorted_indices[:top_n]
        
        final_chunks = []
        for idx, score in top_sorted:
            chunk = self.indexer.chunks[idx].copy()
            chunk['rrf_score'] = score
            final_chunks.append(chunk)

        return final_chunks

    def generate_answer(self, query, context_chunks):
        context_text = "\n".join([c['text'] for c in context_chunks])
        input_text = f"Question: {query}\nContext: {context_text}\nAnswer:"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200, num_beams=4, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def process_query(self, query):
        dense_res, dense_idx = self.dense_retrieval(query)
        sparse_res, sparse_idx = self.sparse_retrieval(query)
        
        fused_chunks = self.rrf_fusion(dense_idx, sparse_idx, top_n=5)
        answer = self.generate_answer(query, fused_chunks)
        
        return {
            "answer": answer,
            "context": fused_chunks,
            "dense_top": dense_res,
            "sparse_top": sparse_res
        }
