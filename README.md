# RAG Demo: PDF-Based Question Answering

## 📌 Approach to Implementing RAG

1. **Document Loading**: We use `PyPDFLoader` from `langchain-community` to extract and load text from the provided PDF document.

2. **Embedding Creation**: The document is converted into vector representations using **Hugging Face's Sentence Transformers** model (`all-MiniLM-L6-v2`).

3. **Vector Store**: We use **FAISS (Facebook AI Similarity Search)** to store and efficiently retrieve the document embeddings.

4. **Retrieval**: When a user asks a question, the system performs a similarity search to find the most relevant PDF sections.

5. **Generation**: We use Hugging Face's \`\` model for natural language generation to formulate responses based on the retrieved content.

6. **Pipeline Flow**:

   - Input: User's question
   - Retrieval: Search FAISS for the most similar PDF passages
   - Generation: Use the retrieved content as context to generate a relevant answer

## 🛠️ Tools and Models Selected

1. **Libraries**:

   - `transformers`: For accessing pre-trained language models (Hugging Face pipeline).
   - `langchain-community`: For document loading and interaction with FAISS.
   - `faiss-cpu`: For fast and scalable vector search.
   - `sentence-transformers`: For creating text embeddings.
   - `pypdf`: For PDF extraction.

2. **Models**:

   - **Embedding Model**: `all-MiniLM-L6-v2`
     - Small and efficient transformer model optimized for semantic search.
   - **Generative Model**: `google/flan-t5-base`
     - Open-source and lightweight model fine-tuned for text-to-text tasks, including question answering.

## 📊 Reasons for Model and Tool Choices

1. **Open-Source and Free**: Avoids API rate limits by relying on Hugging Face's free, public models.

2. **Efficiency**:

   - `all-MiniLM-L6-v2` provides a fast and accurate way to embed large documents.
   - `google/flan-t5-base` is a balanced model in terms of performance and computational cost.

3. **Scalability**: FAISS allows us to perform efficient similarity searches, enabling quick retrieval from large documents.

4. **Flexibility**: This solution is adaptable to other documents and can switch to other models if needed.

## 🚀 How to Run the Notebook

1. Ensure you have installed the required packages:

   ```bash
   !pip install transformers langchain-community langchain pypdf sentence-transformers faiss-cpu
   ```

2. Place the PDF (`Curated_Learning_Resources.pdf`) in the working directory.

3. Execute the cells in the provided Jupyter notebook.

4. Input your question and receive answers based on the PDF content.

