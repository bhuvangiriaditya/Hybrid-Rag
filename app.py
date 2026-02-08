import streamlit as st
import time
from data_loader import DataLoader
from indexer import Indexer
from rag_engine import RAGEngine
from eval_dashboard import render_eval_dashboard

st.set_page_config(page_title="Hybrid RAG System", layout="wide")

@st.cache_resource
def load_system():
    data_loader = DataLoader() # Assuming data is present or will be mocked
    indexer = Indexer()
    rag = RAGEngine(indexer)
    return data_loader, indexer, rag

data_loader, indexer, rag = load_system()
if "data_loader" not in st.session_state:
    st.session_state["data_loader"] = data_loader
if "indexer" not in st.session_state:
    st.session_state["indexer"] = indexer
if "rag_engine" not in st.session_state:
    st.session_state["rag_engine"] = rag

st.title("📚 Hybrid RAG System (Dense + Sparse + RRF)")

# Sidebar for Indexing Control
with st.sidebar:
    st.header("📌 Navigation")
    page = st.radio("Page", ["Chat", "Evaluation"])
    st.markdown("---")
    st.header("🗂️ Knowledge Base")
    if st.button("Reload & Re-Index (500 Articles)"):
        with st.spinner("Fetching data and Indexing..."):
            documents = data_loader.load_all_data() # 200 fixed + 300 random
            num_chunks = indexer.build_index(documents)
        st.success(f"Indexed {len(documents)} articles ({num_chunks} chunks).")
    
    st.markdown("---")
    st.markdown("**System Specs:**")
    st.markdown("- **Dense:** all-MiniLM-L6-v2 + FAISS")
    st.markdown("- **Sparse:** BM25")
    st.markdown("- **Fusion:** Reciprocal Rank Fusion (k=60)")
    st.markdown("- **LLM:** google/flan-t5-base")

if page == "Chat":
    # Main Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Query Input
    query = st.chat_input("Ask a question about the Wikipedia articles...")

    if query:
        if indexer.faiss_index is None:
            st.warning("Please build the index first using the sidebar button!")
        else:
            # Display User Query
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Process Query
            start_time = time.time()
            with st.spinner("Retrieving and Generating..."):
                result = rag.process_query(query)
            end_time = time.time()
            latency = end_time - start_time

            response_text = result['answer']
            
            # Display Assistant Response
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
                st.caption(f"⏱️ Response time: {latency:.2f}s")
                
                # Retrieval details expander
                with st.expander("🔍 Retrieval Details (Top 5 RRF)"):
                    for i, chunk in enumerate(result['context']):
                        chunk_id = chunk.get("chunk_id") or chunk.get("id", "unknown")
                        st.markdown(f"**{i+1}. {chunk['title']}** (`{chunk_id}`) (Score: {chunk['rrf_score']:.4f})")
                        st.markdown(f"> {chunk['text'][:300]}...")
                        st.markdown(f"[Source]({chunk['url']})")
                        st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Top Dense Matches")
                        for c in result['dense_top']:
                            chunk_id = c.get("chunk_id") or c.get("id", "unknown")
                            snippet = c.get("text", "")[:200]
                            st.markdown(f"- **{c['title']}** (`{chunk_id}`)")
                            st.markdown(f"> {snippet}...")
                    with col2:
                        st.markdown("#### Top Sparse Matches")
                        for c in result['sparse_top']:
                            chunk_id = c.get("chunk_id") or c.get("id", "unknown")
                            snippet = c.get("text", "")[:200]
                            st.markdown(f"- **{c['title']}** (`{chunk_id}`)")
                            st.markdown(f"> {snippet}...")
else:
    render_eval_dashboard()
