import os
import tempfile
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once
embedding_model = SentenceTransformer("Tushar0505/fine-tuned-legal-bert")

# Global storage for knowledge base
documents = []
doc_embeddings = None


# ----------------------------
# Chunking utility
# ----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for better retrieval.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ----------------------------
# Document Processing
# ----------------------------
def process_file(file):
    """
    Extract text from uploaded file (PDF, TXT, CSV).
    """
    text = ""
    if file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        reader = PdfReader(tmp_file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        os.remove(tmp_file_path)

    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")

    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        text = df.to_string()

    return text


# ----------------------------
# Add Documents to Knowledge Base
# ----------------------------
def add_documents(uploaded_files):
    """
    Add uploaded files to knowledge base and update embeddings.
    """
    global documents, doc_embeddings

    for file in uploaded_files:
        raw_text = process_file(file)
        if raw_text.strip():
            chunks = chunk_text(raw_text)
            documents.extend(chunks)

    if documents:
        doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)


# ----------------------------
# Retrieve Context for Query
# ----------------------------
def embed_and_retrieve(query, top_k=3):
    """
    Retrieve most relevant chunks for a query using cosine similarity.
    """
    global documents, doc_embeddings
    if not documents or doc_embeddings is None:
        return ""  # no documents in memory

    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_chunks = [documents[i] for i in top_indices]

    return "\n".join(retrieved_chunks)