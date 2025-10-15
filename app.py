
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Custom CSS for Styling ---
def load_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
        body, .stApp {
            background-color: #BADFDB !important; font-family: 'Times New Roman', serif !important; color: #1F2937;
        }
        h1, h2, h3 { color: #0B5345; }
        .stButton > button {
            border-radius: 20px; border: 2px solid #0B5345; background-color: #0B5345;
            color: white !important; padding: 10px 24px; font-weight: bold;
            font-family: 'Times New Roman', serif; transition: all 0.3s;
        }
        .stButton > button:hover { background-color: #117A65; border-color: #117A65; }
        section[data-testid="stFileUploader"] button {
            background-color: #0B5345; color: white; border-radius: 20px;
        }
        section[data-testid="stFileUploader"] button:hover { background-color: #117A65; }
        .stTextInput > label, .stFileUploader > label { font-size: 1.2em; font-weight: bold; color: #0B5345; }
        .answer-box {
            background-color: #F0FDF4; border-left: 6px solid #117A65; padding: 15px;
            border-radius: 8px; margin-top: 20px; font-size: 1.1em; color: #1F2937;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Backend RAG Logic ---
def ingest_and_process_docs(file_paths: list[str]):
    documents = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"): loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"): loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"): loader = TextLoader(file_path)
        else: continue
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, persist_directory="./db")
    vectorstore.persist()

def query_knowledge_base(user_query: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    prompt_template = "Using these documents, answer the user's question succinctly: {question}\n\nContext: {context}"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # ***** ✅ THE FINAL FIX IS HERE: We've added your project ID *****
    project_id = "497766208599"
    llm = ChatVertexAI(
        project=project_id,
        model_name="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain.invoke(user_query)

# --- Frontend UI ---
st.set_page_config(page_title="InsightEngine", layout="centered")
load_css()
st.title("💡 InsightEngine")
st.markdown("---")
st.header("Upload your files")
uploaded_files = st.file_uploader( "Upload PDF, DOCX, or TXT documents", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
if st.button("Process Documents") and uploaded_files:
    with st.spinner("Reading and storing document knowledge..."):
        file_paths = []
        for file in uploaded_files:
            with open(file.name, "wb") as f: f.write(file.getbuffer())
            file_paths.append(file.name)
        ingest_and_process_docs(file_paths)
        st.success("Documents processed successfully!")
        for path in file_paths: os.remove(path)
st.markdown("---")
st.header("Ask your question")
user_query = st.text_input("Enter your question based on the documents you uploaded")
if user_query:
    with st.spinner("Searching for the answer..."):
        answer = query_knowledge_base(user_query)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
