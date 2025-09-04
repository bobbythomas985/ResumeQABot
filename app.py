import os
import gradio as gr
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key")  # Replace locally if needed
client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf(file):
    raw_text = extract_text_from_pdf(file)
    chunks = split_into_chunks(raw_text)
    embeddings = embedding_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return "PDF processed successfully.", (index, chunks)

def query_document(question, state):
    index, chunks = state
    if index is None or not chunks:
        return "Please upload and process a PDF first."

    query_vector = embedding_model.encode([question])
    distances, indices = index.search(np.array(query_vector), k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes and answers questions about documents."},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error from Groq API: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Resume Q&A Bot (Groq + FAISS)")
    state = gr.State((None, None))

    with gr.Row():
        file_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")
    status_output = gr.Textbox(label="Status")

    with gr.Row():
        question = gr.Textbox(label="Enter your question")
        ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=10)

    upload_btn.click(process_pdf, inputs=file_input, outputs=[status_output, state])
    ask_btn.click(query_document, inputs=[question, state], outputs=answer_output)

demo.launch()
