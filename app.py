import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_pipeline import extract_text_from_pdf, chunk_text, create_vector_store, retrieve, generate_prompt, call_llm


# ==============================
# Globals
# ==============================
#embedder = SentenceTransformer("all-MiniLM-L6-v2")
vector_store_global = None
chunks_global = None

# ==============================
# Gradio Functions
# ==============================
def load_pdfs(files):
    global vector_store_global, chunks_global
    all_text = ""
    for file in files:
        all_text += extract_text_from_pdf(file) + "\n"
    chunks = chunk_text(all_text)
    index, _ = create_vector_store(chunks)
    chunks_global = chunks
    vector_store_global = index
    return "PDFs loaded and indexed successfully!"

def chat_with_rag(user_message):
    if vector_store_global is None or chunks_global is None:
        return "Please upload and load PDFs first."
    context = retrieve(user_message, chunks_global, vector_store_global, top_k=5)
    prompt = generate_prompt(context,user_message)
    response = call_llm(prompt)
    return response

# ==============================
# Gradio UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("# PDF Question Answering with RAG")

    with gr.Row():
        file_input = gr.File(
            label="Upload PDF(s)",
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath"
        )
        load_button = gr.Button("Load PDFs")

    status_output = gr.Textbox(label="Status")

    with gr.Row():
        user_message = gr.Textbox(label="Ask a question", lines=2)
        send_button = gr.Button("Send")

    chat_output = gr.Textbox(label="Answer", lines=10)

    load_button.click(fn=load_pdfs, inputs=file_input, outputs=status_output)
    send_button.click(fn=chat_with_rag, inputs=user_message, outputs=chat_output)

if __name__ == "__main__":
    demo.launch(share=True)
