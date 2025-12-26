import streamlit as st
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# ====== Your RAG imports ======
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import numpy as np
from langchain_groq import ChatGroq

# ====== ENV ======
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIR = "data/pdf"
VECTOR_DIR = "data/vector_store"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ====== Streamlit Config ======
st.set_page_config(page_title="PDF RAG Q&A", layout="wide")
st.title("PDF Question Answering using RAG + LLM")

# =====================================================
# ----------------- RAG COMPONENTS --------------------
# =====================================================

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    def __init__(self, persist_directory=VECTOR_DIR, collection_name="pdf_docs"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents, embeddings):
        ids = [f"doc_{uuid.uuid4().hex}" for _ in documents]
        self.collection.add(
            ids=ids,
            documents=[d.page_content for d in documents],
            metadatas=[d.metadata for d in documents],
            embeddings=embeddings.tolist()
        )


class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5, score_threshold=0.2):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist
            if score >= score_threshold:
                docs.append({
                    "content": doc,
                    "metadata": meta,
                    "score": score
                })
        return docs


def rag_answer(question, retriever, llm):
    docs = retriever.retrieve(question)
    if not docs:
        return "No relevant context found.", [], 0.0

    context = "\n\n".join([d["content"] for d in docs])
    confidence = max(d["score"] for d in docs)

    prompt = f"""
Use the context below to answer concisely.

Context:
{context}

Question: {question}
Answer:
"""

    response = llm.invoke(prompt)
    return response.content, docs, confidence

# =====================================================
# ----------------- STREAMLIT CACHE -------------------
# =====================================================

@st.cache_resource
def load_models():
    embedder = EmbeddingManager()
    vectorstore = VectorStore()
    retriever = RAGRetriever(vectorstore, embedder)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    return embedder, vectorstore, retriever, llm


embedder, vectorstore, retriever, llm = load_models()

# =====================================================
# ----------------- PDF UPLOAD ------------------------
# =====================================================

st.sidebar.header(" Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs..."):
            all_docs = []
            for file in uploaded_files:
                path = os.path.join(DATA_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source_file"] = file.name
                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)
            embeddings = embedder.generate_embeddings(
                [c.page_content for c in chunks]
            )
            vectorstore.add_documents(chunks, embeddings)

        st.success(f"Processed {len(uploaded_files)} PDFs successfully!")

# =====================================================
# ----------------- QUESTION UI -----------------------
# =====================================================

st.subheader("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer, sources, confidence = rag_answer(question, retriever, llm)

        st.markdown("###  Answer")
        st.write(answer)

        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        if sources:
            st.markdown("### 📚 Sources")
            for i, src in enumerate(sources, 1):
                st.markdown(
                    f"**{i}. {src['metadata'].get('source_file','')}**  \n"
                    f"Score: `{src['score']:.2f}`"
                )
