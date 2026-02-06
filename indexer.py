from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Indexer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384 # Dimension for all-MiniLM-L6-v2
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = [] # Store chunk metadata: {text, url, title, id}
        self.tokenized_corpus = []

    def chunk_text(self, text, chunk_size=200, overlap=50):
        """
        Simple word-based chunker.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def build_index(self, documents):
        """
        documents: list of dict {url, title, text}
        """
        self.chunks = []
        self.tokenized_corpus = []
        
        # 1. Chunking
        print("Chunking documents...")
        for doc in documents:
            doc_chunks = self.chunk_text(doc['text'])
            for i, chunk_text in enumerate(doc_chunks):
                self.chunks.append({
                    "id": f"{doc['title']}_{i}",
                    "title": doc['title'],
                    "url": doc['url'],
                    "text": chunk_text
                })
                # For BM25
                self.tokenized_corpus.append(chunk_text.lower().split())

        # 2. Build Sparse Index (BM25)
        print("Building BM25 Index...")
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        # 3. Build Dense Index (FAISS)
        print("Building FAISS Index...")
        embeddings = self.embedding_model.encode([c['text'] for c in self.chunks])
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        print(f"Indexing complete. {len(self.chunks)} chunks indexed.")
        return len(self.chunks)
