import os
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

import uvicorn # Added for running the FastAPI app

# --- Configuration (read from environment variables) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi3:mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
DATA_DIR = "/app/data"
CHROMA_DB_DIR = "/app/chroma_db"

# FastAPI application instance
app = FastAPI(
    title="RAG Docker App",
    description="Retrieval-Augmented Generation (RAG) service with Ollama and mixed data sources.",
    version="1.0.0"
)

# Global variable to hold the RAG chain
qa_chain = None

# Pydantic model for the query request body
class QueryRequest(BaseModel):
    query: str

# Pydantic model for the response
class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# --- Helper function to load documents ---
def load_documents_from_dir(directory: str):
    documents = []
    data_path = Path(directory)

    if not data_path.exists():
        print(f"Warning: Data directory {data_path} does not exist. No documents will be loaded.")
        return []

    print(f"Loading documents from {data_path}...")
    for file_path in data_path.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
                print(f"  Loaded PDF: {file_path.name}")
            elif file_path.suffix.lower() in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents.extend(loader.load())
                print(f"  Loaded Markdown: {file_path.name}")
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path))
                documents.extend(loader.load())
                print(f"  Loaded Text: {file_path.name}")
            else:
                print(f"  Skipping unsupported file type: {file_path.name}")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
    return documents

# --- RAG system setup function ---
async def setup_rag_system():
    global qa_chain # Declare global to modify the global variable

    print("Starting RAG system setup...")
    # 1. Load Documents
    all_docs = load_documents_from_dir(DATA_DIR)

    if not all_docs:
        print("No documents loaded. Please place your data files in the 'data/' directory.")
        # We won't exit, but the RAG chain will be None or handle empty docs
        # If you want to force exit, raise an exception here
        # raise RuntimeError("No documents found to build RAG system.")

    # 2. Split Documents into Chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Loaded {len(all_docs)} documents and split into {len(chunks)} chunks.")

    # 3. Initialize Embedding Model
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = None
    try:
        if "ollama" in EMBEDDING_MODEL_NAME.lower() or "nomic" in EMBEDDING_MODEL_NAME.lower():
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_HOST)
            _ = embeddings.embed_query("test embedding") # Test if Ollama embedding works
        else:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            _ = embeddings.embed_query("test embedding") # Test if HuggingFace embedding works
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model {EMBEDDING_MODEL_NAME}: {e}")
        print("Please ensure the embedding model is available (e.g., pulled via Ollama or downloaded for HuggingFace).")
        # If embedding model fails, we cannot proceed with RAG
        raise HTTPException(status_code=500, detail=f"Failed to initialize embedding model: {e}")

    # 4. Vector Database Storage (ChromaDB)
    print("Setting up vector store...")
    if Path(CHROMA_DB_DIR).exists() and any(Path(CHROMA_DB_DIR).iterdir()):
        print("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new vector store (this might take a while for large datasets)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()
        print("Vector store created and persisted.")

    # 5. Initialize Ollama LLM
    print(f"Initializing Ollama LLM: {OLLAMA_MODEL_NAME} from {OLLAMA_HOST}...")
    try:
        llm = Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_HOST)
        _ = llm.invoke("Hi") # Test if LLM responds
        print("Ollama LLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama LLM {OLLAMA_MODEL_NAME}: {e}")
        print("Please ensure the LLM is pulled via Ollama and Ollama container is running.")
        # If LLM fails, we cannot proceed with RAG
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {e}")

    # 6. Create RAG Chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG system setup complete.")

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI startup event triggered.")
    await setup_rag_system()
    print("FastAPI application ready to serve requests.")

# --- FastAPI Endpoints ---
@app.get("/")
async def read_root():
    """
    Health check endpoint.
    """
    return {"message": "RAG Docker App is running!"}

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """
    Endpoint to query the RAG application.
    """
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please wait or check logs.")

    print(f"Received query: {request.query}")
    try:
        result = qa_chain.invoke({"query": request.query})
        
        # Format source documents for the response model
        formatted_sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                formatted_sources.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return RAGResponse(
            answer=result["result"],
            sources=formatted_sources
        )
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# --- Run the FastAPI app ---
if __name__ == "__main__":
    # When running directly (e.g., for local testing without Docker Compose),
    # this will also trigger the startup event.
    uvicorn.run(app, host="0.0.0.0", port=8000)