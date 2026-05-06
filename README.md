# 📄 AI Document Assistant (RAG-based)

An AI-powered document assistant that allows users to upload PDF/TXT files and ask questions using natural language. The system uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers grounded in document content.

---

## 🚀 Features

- 📂 Upload multiple PDF and TXT documents
- 🔍 Semantic search using embeddings
- 🤖 Context-aware Q&A using LLMs
- 💬 Chat-based interface with session memory
- ⚡ Fast retrieval using vector search
- 🎯 Reduced hallucination via grounded responses
- 🔄 Multi-document support with unified querying

---

## 🧠 Tech Stack

- **Frontend:** Streamlit  
- **Backend / Orchestration:** LangChain, LangGraph  
- **LLM:** OpenAI (GPT-4o-mini)  
- **Embeddings:** OpenAI `text-embedding-3-large`  
- **Vector Store:** InMemoryVectorStore  
- **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter  
- **State Management:** Streamlit session state  

---

## ⚙️ How It Works

### 1. Document Upload
Users upload PDF or TXT files via the UI.

### 2. Parsing & Chunking
Documents are split into smaller chunks for better retrieval:
- `chunk_size = 1000`
- `chunk_overlap = 200`

### 3. Embedding Generation
Each chunk is converted into vector embeddings using OpenAI.

### 4. Vector Storage
Embeddings are stored in an in-memory vector database.

### 5. Query Processing
- User query is converted into embedding
- Top-k similar chunks are retrieved

### 6. Response Generation
- Retrieved context + query → LLM → final answer
- Responses are grounded strictly in document content

---

## 🏗️ Architecture
User Query
↓
Embedding Generation
↓
Vector DB (Similarity Search)
↓
Top-K Relevant Chunks
↓
LLM (GPT-4o-mini)
↓
Final Answer (Grounded Response)

---

## 📁 Project Structure


.
├── app.py # Main Streamlit app
├── doc_files/ # Uploaded documents storage
├── .env # Environment variables
├── requirements.txt # Dependencies
└── README.md

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ai-document-assistant.git
cd ai-document-assistant
```
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key

## ▶️ Run the App
streamlit run app.py

## 👨‍💻 Author

Abhinav Tomar
