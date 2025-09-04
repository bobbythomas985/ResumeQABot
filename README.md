# 📄 Resume / Document Q&A Bot (Groq + FAISS)

An interactive **AI-powered document Q&A app** built with **Gradio**, **FAISS**, and **Groq's LLM API**.  
Upload a PDF, process it into vector embeddings, and ask natural language questions — get accurate, context-aware answers instantly.

---

## 🚀 Features

✅ **PDF Text Extraction** – Reads and processes any PDF file using `PyPDF2`  
✅ **Chunking & Embedding** – Splits text into 500-word chunks and encodes using `SentenceTransformer`  
✅ **Vector Search with FAISS** – Efficient similarity search for relevant chunks  
✅ **Groq-Powered Q&A** – Sends context + query to `llama3-8b-8192` for natural answers  
✅ **Interactive Web UI** – Clean Gradio interface for PDF upload & question answering  

---

## 🛠 Tech Stack

- **Python 3.9+**
- [Gradio](https://gradio.app/) – UI framework
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [SentenceTransformers](https://www.sbert.net/) – Text embeddings (`all-MiniLM-L6-v2`)
- [Groq API](https://groq.com/) – LLM for answering questions
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing
- **NumPy** – Array handling

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/document-qa-bot.git
cd document-qa-bot
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Set Up Your API Key
Export your Groq API key as an environment variable:
####Linux / macOS
```bash
export GROQ_API_KEY="your_api_key_here"
```
####Windows
```powershell
setx GROQ_API_KEY "your_api_key_here"
```
Alternatively, replace the placeholder in app.py:
```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key")
```
### 4️⃣ Run the App
```bash
python app.py
```

---

## 🖥️ How It Works

1️⃣ **Upload a PDF**  
📄 Select any PDF file you want to analyze.

2️⃣ **Process PDF**  
🛠 Extracts text, splits it into 500-word chunks, embeds each chunk with `SentenceTransformer`, and builds a FAISS vector index for fast retrieval.

3️⃣ **Ask a Question**  
❓ Converts your question into a vector, searches for the **top 3 most relevant chunks**, and prepares a context window.

4️⃣ **Groq API Call**  
🤖 Sends context + your question to `llama3-8b-8192` model via Groq API → returns a **natural language answer**.

---

## 🔮 Future Improvements

- 🔍 **Support for multiple PDFs / knowledge bases** – Combine several documents into a single searchable index.  
- 📑 **Source citation highlighting** – Show exactly which page/chunk was used for the answer.  
- 🚀 **Deploy to Hugging Face Spaces** – Make it easily shareable online with just a browser link.  
- 🌐 **Multilingual question support** – Integrate Google Translate API for cross-language Q&A.

---
