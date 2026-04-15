import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
st.set_page_config(page_title="📄 AI Document Assistant", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DOC_PATH = os.path.join(BASE_DIR, "doc_files")
os.makedirs(DOC_PATH, exist_ok=True)


if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files" not in st.session_state:
    st.session_state.files = []


st.title("📄 AI Document Assistant")
st.markdown("Upload documents and ask questions from them instantly.")


with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.files = uploaded_files

    if st.session_state.files:
        st.success(f"{len(st.session_state.files)} file(s) uploaded")

        if st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                all_docs = []

                for file in st.session_state.files:
                    file_path = os.path.join(DOC_PATH, file.name)

                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())

                    # Load documents
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        all_docs.extend(docs)

                    elif file.name.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        all_docs.append({"page_content": text})

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = splitter.split_documents(all_docs)

                # Embeddings
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                vector_db = InMemoryVectorStore.from_documents(docs, embeddings)

                # LLM
                llm = ChatOpenAI(model="gpt-4o-mini")

                @tool
                def retrieve_context(query: str):
                    """Retrieve relevant document chunks."""
                    results = vector_db.similarity_search(query, k=3)
                    context = ""
                    for doc in results:
                        context += doc.page_content + "\n\n"
                    return context

                # Agent
                agent = create_agent(
                    model=llm,
                    tools=[retrieve_context],
                    system_prompt="Answer using document context only.",
                    checkpointer=InMemorySaver()
                )

                st.session_state.agent = agent
                st.success("✅ Documents processed successfully!")


if st.session_state.agent:
    st.subheader("💬 Ask Questions")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": 1}}
                )

                answer = response["messages"][-1].content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload and process documents to start chatting.")