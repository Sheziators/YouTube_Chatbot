import streamlit as st
import os
from dotenv import load_dotenv

# =========================
# 🔑 SET API KEY
# =========================

load_dotenv()

os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# =========================
# 📦 IMPORTS
# =========================
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# =========================
# 🎯 PAGE CONFIG
# =========================
st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

st.title("🎥 YouTube RAG Chatbot (HuggingFace)")
st.write("Chat with any YouTube video using AI")

# =========================
# 🤖 LOAD LLM
# =========================
@st.cache_resource
def load_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        task="conversational",
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
    return ChatHuggingFace(llm=llm_endpoint)

llm = load_llm()

# =========================
# 🧠 MEMORY
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

def get_session_history(session_id):
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_history[session_id]

# =========================
# 🎥 FETCH TRANSCRIPT
# =========================
def get_transcript(video_id):
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_list = fetched.to_raw_data()
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# =========================
# ✂️ SPLIT TEXT
# =========================
def split_docs(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.create_documents([transcript])

# =========================
# 🧠 VECTOR STORE
# =========================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vectorstore(documents):
    embeddings = load_embeddings()
    return FAISS.from_documents(documents, embeddings)

# =========================
# 🔗 BUILD RAG CHAIN
# =========================
def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.
If the context is insufficient, just say "I don't know based on the video."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
    "context": (
        RunnableLambda(lambda x: x["question"])  # extract string
        | retriever
        | RunnableLambda(format_docs)
    ),
    "question": RunnableLambda(lambda x: x["question"])
})

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser

    chain_with_memory = RunnableWithMessageHistory(
        main_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return chain_with_memory

# =========================
# 🎥 SIDEBAR
# =========================
st.sidebar.header("🔗 YouTube Setup")

video_id = st.sidebar.text_input("Enter YouTube Video ID")

if video_id:
    st.sidebar.video(f"https://www.youtube.com/watch?v={video_id}")

if st.sidebar.button("Load Video"):
    with st.spinner("Processing video..."):
        transcript = get_transcript(video_id)

        if transcript:
            docs = split_docs(transcript)
            vectorstore = create_vectorstore(docs)
            st.session_state.chain = build_chain(vectorstore)
            st.success("✅ Video Loaded!")

# =========================
# 💬 CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask a question about the video...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if "chain" in st.session_state:
                result = st.session_state.chain.invoke(
                    {"question": query},
                    config={"configurable": {"session_id": "user1"}}
                )
                answer = result
            else:
                answer = "⚠️ Please load a YouTube video first."

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
