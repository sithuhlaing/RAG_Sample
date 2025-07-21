# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LangChain RAG with Hugging Face Inference API",
    description="A FastAPI application demonstrating RAG using LangChain, Hugging Face LLM, and Hugging Face Embeddings.",
    version="1.0.0"
)

# --- Global variables for RAG components ---
# These will be initialized once on startup
qa_chain = None
embeddings_model = None
llm_model = None
vectorstore_instance = None

# --- Configuration (from Environment Variables) ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Ensure HF_TOKEN is set. In a real app, you might want more robust error handling
# or a startup check. For now, we'll raise an error if not found.
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running the application.")

# Hugging Face model IDs
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta" # Or 'mistralai/Mistral-7B-Instruct-v0.2' etc.

# --- Pydantic Models for API Request/Response ---
class QueryRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class SourceDocument(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]

# --- RAG Initialization Function ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize RAG components on application startup.
    This ensures models and vector stores are loaded once.
    """
    global qa_chain, embeddings_model, llm_model, vectorstore_instance

    print("Initializing Hugging Face Embeddings model...")
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name=EMBEDDING_MODEL_ID
    )
    print(f"Embedding model '{EMBEDDING_MODEL_ID}' loaded.")

    print("Initializing Hugging Face LLM model...")
    llm_model = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_ID,
        temperature=0.7,
        max_new_tokens=500,
        huggingfacehub_api_token=HF_TOKEN
    )
    print(f"LLM model '{LLM_MODEL_ID}' loaded.")

    print("Loading documents...")
    try:
        loader = TextLoader("./data/state_of_the_union.txt")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to load knowledge base documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating vector store (this may take a moment as embeddings are generated)...")
    try:
        vectorstore_instance = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            collection_name="state-of-the-union-rag"
        )
        print("Vector store created.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vector store with embeddings.")

    print("Setting up RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vectorstore_instance.as_retriever(),
        return_source_documents=True
    )
    print("RAG chain initialized successfully!")

# --- API Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Process a user query using the RAG system and return the answer along with source documents.
    """
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized. Please wait a moment.")

    print(f"Received query: {request.query}")
    try:
        # Dynamically set LLM parameters if provided in the request
        # Note: This requires re-initializing the LLM or updating its parameters,
        # which can be complex for HuggingFaceEndpoint. For simplicity,
        # we'll use the pre-initialized LLM's parameters.
        # If you need dynamic LLM parameters per request, you might need to
        # create a new HuggingFaceEndpoint instance inside this function,
        # which could impact performance.
        # For this example, we'll stick to the startup-configured LLM.

        response = qa_chain.invoke({"query": request.query})

        source_docs = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in response["source_documents"]
        ]

        return QueryResponse(answer=response["result"], source_documents=source_docs)
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.get("/")
async def read_root():
    return {"message": "LangChain RAG FastAPI is running! Use /query endpoint for RAG."}