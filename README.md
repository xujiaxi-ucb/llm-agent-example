# ETF Fact-Sheet QA & Comparator

This project provides a system for querying and comparing ETF (Exchange Traded Fund) fact sheets using a combination of LangGraph, Pinecone, and Gradio.

## Features

- **Single PDF QA**: Upload a single ETF fact sheet (PDF) and ask questions to extract key information.
- **Compare Two PDFs**: Upload two ETF fact sheets (PDFs) and compare key metrics extracted from both.
- **RAG-based Answers**: Utilizes Retrieval-Augmented Generation (RAG) to provide answers with citations.
- **Structured Data Extraction**: Extracts key ETF metrics into a structured JSON format.

## Architecture

- **Design**: The system involves document chunking, vectorization into Pinecone (serverless), metadata-filtered retrieval, and generation of cited answers. RAGAS is used for evaluation.
- **Indexing**: Dimension and metric (cosine/euclidean/dotproduct) are selected based on the embedding model. Serverless index creation/management, filtering, and hybrid retrieval follow official Pinecone guidelines.
- **Orchestration**: LangGraph is used for node-based state machine orchestration. For complex scenarios, LCEL Runnables can be used for parallel/streaming operations.
- **Frontend**: The user interface is built with Gradio Blocks.
- **Extensibility**: Designed for MCP (Model Context Protocol) tool output, allowing direct connection by IDEs, desktop applications, or other agents.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/xujiaxi-ucb/llm-agent-example.git
    cd llm-agent-example
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    Create a `.env` file in the root directory and add your API keys and configurations:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="your_pinecone_environment" # e.g., us-east-1
    PINECONE_INDEX="finflow" # or your preferred index name
    EMBED_MODEL="text-embedding-3-small" # or other embedding model
    GEN_MODEL="gpt-5-nano" # or other generation model
    ```
4.  **Install `poppler-utils`**:
    This is required for `pdftotext` which is used as a fallback for PDF text extraction.
    -   **Debian/Ubuntu**: `sudo apt-get install poppler-utils`
    -   **macOS**: `brew install poppler`
    -   **Windows**: Download from [here](https://poppler.freedesktop.org/) and add to PATH.

## Usage

To run the Gradio application:

```bash
python app/etf_app.py
```

This will launch a local Gradio interface in your browser, where you can interact with the ETF QA and comparison features.

## Project Structure

-   `app/etf_app.py`: Gradio application interface.
-   `graph/etf_pipeline.py`: LangGraph pipeline for RAG and metric extraction.
-   `vec/pinecone_store.py`: Pinecone vector store operations (upsert, query, index management).
-   `data/`: Contains sample PDF fact sheets (`spy.pdf`, `voo.pdf`).
-   `requirements.txt`: Python dependencies.
