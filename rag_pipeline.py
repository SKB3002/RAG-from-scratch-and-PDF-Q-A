import fitz
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests, os

# --- Step 1: PDF Extraction (cleaner text) ---
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # "text" ensures structured text
    # Remove excessive newlines / spaces
    text = re.sub(r'\n+', ' ', text)  # collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)  # collapse spaces
    return text.strip()


# --- Step 2: Chunking (semantic) ---
def chunk_text(text: str, chunk_size=100):
    """
    Chunk text into overlapping segments so sentences aren't cut randomly.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# --- Step 3: Embeddings + FAISS ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve(query, chunks, index, top_k=5):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb), k=top_k)
    return [chunks[i] for i in I[0]]


def generate_prompt(context,query):
    prompt = f"""
    You are a PDF-based assistant. Answer strictly using the provided context.
    If the answer is not in the context, reply with exactly:
    "The document does not contain this information."

    Context:
    {context}

    Question:
    {query}
    """
    return prompt

# --- Step 4: LLM Call via Groq ---
os.environ["GROQ_API_KEY"] = "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"

def call_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}

    payload = {
        #"model": "llama3-70b-instruct", 
        "model": "llama3-8b-8192",# best Groq free model 
        "messages": [
            {"role": "system", "content": "You are a precise assistant. Avoid irrelevant text. Answer in clean sentences."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    	"max_tokens": 500
    }

    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()["choices"][0]["message"]["content"]
