# 🎥 YouTube RAG Chatbot

An AI-powered Streamlit application that allows users to **chat with any YouTube video** using a Retrieval-Augmented Generation (RAG) pipeline. The system extracts video transcripts, processes them into embeddings, and uses a Large Language Model to generate accurate, context-aware answers.

---

## 🚀 Features

- 🎥 Chat with any YouTube video using its transcript  
- 🧠 RAG-based architecture for accurate, context-aware responses  
- 🔎 Semantic search using FAISS vector database  
- 🤖 Hugging Face LLM integration (LLaMA / Mistral supported)  
- 💬 Conversation memory for contextual chat  
- ⚡ Fast and interactive UI built with Streamlit  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend/NLP:** LangChain  
- **LLM:** Hugging Face Transformers  
- **Embeddings:** Sentence Transformers  
- **Vector Database:** FAISS  
- **Data Source:** YouTube Transcript API  

---

## ⚙️ How It Works

1. User enters a YouTube video ID  
2. Transcript is extracted using YouTube Transcript API  
3. Text is split into chunks for processing  
4. Embeddings are generated and stored in FAISS  
5. User asks a question  
6. Relevant chunks are retrieved  
7. LLM generates an answer based only on retrieved context  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/YouTube_Chatbot.git
cd YouTube_Chatbot
