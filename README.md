# RAG-from-scratch-and-PDF-Q-A

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (e.g., job offers, contracts, or policies) and ask natural language questions about them.

Instead of returning raw text chunks, the system uses **FAISS vector search** to retrieve the most relevant content and an **LLM** to generate clear and contextualized answers.

---

## Features
- 📂 **Document Upload:** Easily load job offers, agreements, or any text-based documents.  
- 🔍 **FAISS Retrieval:** Efficient similarity search over embedded document chunks.  
- 🧠 **LLM-Powered Answers:** Generates human-like responses instead of just copy-pasting text.  
- 🎨 **Gradio Interface:** Simple UI for interacting with the system.  
- ☁️ **Deployable Anywhere:** Works locally and on Hugging Face Spaces.  

---

## Tech Stack
- **Python 3.10+**
- **FAISS** – for vector indexing and retrieval
- **Sentence Transformers / OpenAI API** – for embeddings + answer generation
- **Gradio** – for frontend UI
- [**Hugging Face Spaces**]((https://huggingface.co/spaces/SKB3002/RAG_from_scratch_for_PDF_QandA))

---

## 📂 Project Structure

├── app.py # Gradio app (frontend + API interface)

├── rag_pipeline.py # RAG pipeline logic (retriever + generator)

├── requirements.txt # Python dependencies

└── README.md # Project description (this file)

---

## How It Works
1. **Document Upload** – User uploads a text-based file.  
2. **Chunking & Embedding** – Text is split into chunks and converted into embeddings.  
3. **Vector Search** – FAISS retrieves the most relevant chunks for the query.  
4. **LLM Answer Generation** – The retrieved context is passed into the LLM to generate a clear answer.  

---

## Installation & Usage (Local)
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

(https://huggingface.co/spaces/SKB3002/RAG_from_scratch_for_PDF_QandA)
---

## Licence

This project is released under the MIT License.
Feel free to fork, modify, and build on top of it.

👨‍💻 Author: Suyash Bhatkar
🔗 Hugging Face Profile: SKB3002
