# RAG Demo using Streamlit, Hugging Face (for embeddings), and FAISS

import streamlit as st
import fitz  # PyMuPDF for PDF reading
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Sentence Transformer model (free, offline embedding model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast

# PDF Processing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    return text

# Chunking the content (for better retrieval performance)
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings using SentenceTransformer (free and fast)
def get_embedding(text):
    return embedding_model.encode(text)

# Build FAISS index
def create_faiss_index(chunks):
    embeddings = np.array([get_embedding(chunk) for chunk in chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 norm for similarity search
    index.add(embeddings)
    return index, embeddings

# Retrieve relevant context using FAISS
def retrieve_relevant_chunks(query, chunks, index, top_k=3):
    query_embedding = np.array([get_embedding(query)])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer using OpenAI GPT (optional – can switch to open-source models)
def generate_answer(context, question):
    prompt = f"""Use the following information to answer the question:

{context}

Question: {question}
Answer: """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError:
        return "OpenAI quota exceeded. Consider using an alternative model."

# Main Streamlit app
def main():
    st.title("RAG Demo: Job Preparation Resource Assistant")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.success("PDF uploaded successfully!")

        # Extract text and chunk it
        pdf_text = extract_text_from_pdf("uploaded.pdf")
        chunks = chunk_text(pdf_text)
        st.info(f"Extracted {len(chunks)} chunks from the PDF.")

        # Create FAISS index
        index, _ = create_faiss_index(chunks)

        query = st.text_input("Ask a question about the job resources:")

        if st.button("Get Answer") and query:
            context = retrieve_relevant_chunks(query, chunks, index)
            answer = generate_answer("\n".join(context), query)

            st.write("### Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
