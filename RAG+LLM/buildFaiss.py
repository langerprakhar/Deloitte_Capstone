# RAG+LLM/create_vectorstore.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs_dir = "RAG+LLM/Books"
all_docs = []

# Load and split all PDFs in the books/ folder
for filename in os.listdir(docs_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_dir, filename))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)

# Build the vector store
faiss_store = FAISS.from_documents(all_docs, embedding)
faiss_store.save_local("RAG+LLM/faiss_store")
print("✅ FAISS vectorstore created successfully.")
