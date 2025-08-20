# RAG Agent for IT Queries

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** for querying a local knowledge base. The agent can answer user questions using documents ingested into a **FAISS vector store** and can also perform reasoning with relevant context using an LLM.

The system is built using **LangChain**, **Ollama LLM**, and **FAISS embeddings**.

---

## How It Works

### 1. Document Ingestion (`ingest.py`)
The ingestion script:
1. Reads all knowledge base documents.
2. Uses the **Ollama embedding model** (`mxbai-embed-large`) to convert documents into vector representations.
3. Stores the vectors in a **FAISS index** (`faiss_index/`) for fast similarity search.


---

### 2. RAG Agent Setup (`rag_agent_chat.py`)

The chatbot is implemented in two modes:

#### a. React Agent (Primary)
- Uses **LangChain React Agent** for reasoning and tool usage.
- Tools implemented:
  - **knowledge_search**: Searches the local FAISS vector index for relevant documents.
- Workflow:
  1. Receives user input.
  2. Uses the agent to decide whether to search documents or respond directly.
  3. Performs multi-step reasoning using the `Thought → Action → Observation` loop.
  4. Returns the final answer to the user.

#### b. RAG Fallback Chain
- If the React Agent fails, a **simple RAG chain** is used:
  1. Retrieves top-k relevant documents from FAISS.
  2. Provides the documents as context to the LLM (`mistral`).
  3. Generates an answer with `StrOutputParser()`.

---

### 3. Embeddings and Vector Store
- **Embedding Model:** `mxbai-embed-large` (Ollama embeddings)
- **Vector Store:** FAISS, stored locally in `faiss_index/`
- **Retriever:** Returns top-4 most relevant documents for a given query.

