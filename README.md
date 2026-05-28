# Lexicon RAG Engine

A professional-grade, lightweight Retrieval-Augmented Generation (RAG) system engineered from scratch in Python. This system focuses on the programmatic ingestion, chunking, semantic embedding, and local vector storage of long-form textual data (such as books and academic papers).

## 🏗️ Architecture & Data Pipeline

The project implements the foundational phases of a modern RAG pipeline:

1. **Document Ingestion & Chunking**: Raw text is parsed and split into manageable, overlapping text blocks using a `RecursiveCharacterTextSplitter`. This preserves semantic context at the boundaries of each fragment.
2. **Dense Vector Embedding**: Text chunks are converted into 384-dimensional dense vectors using the serverless Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`).
3. **Vector Storage**: High-dimensional embeddings, along with their textual metadata and generated identifiers, are indexed and persisted locally using **ChromaDB** (an open-source, SQLite-backed vector database).

---

## 📁 Project Structure

```text
lexicon-rag/
│
├── chroma_db/          # Persistent local database directory (SQLite binary + index)
├── .venv/              # Isolated Python virtual environment
├── .env                # Private infrastructure and API credentials (git-ignored)
├── .gitignore          # Version control exclusion policies
├── main.py             # Primary ETL pipeline execution script
├── README.md           # System documentation
└── requirements.txt    # Project dependencies
🛠️ System Requirements
Runtime: Python 3.10 or higher

Credentials: Hugging Face Fine-grained Access Token (with Inference permissions enabled)

Operating System: macOS / Linux / Windows

🚀 Installation & Setup
Follow these steps to establish the isolated runtime environment and execute the pipeline:

1. Environment Isolation
Initialize a clean virtual environment to prevent dependency conflicts with your global Python installation:

Bash
python3 -m venv .venv
Activate the virtual environment context (macOS/Linux):

Bash
source .venv/bin/activate
2. Dependency Ingestion
Install the required software engineering and machine learning toolkit via pip:

Bash
pip install -r requirements.txt
3. Secrets Configuration
Create a .env file in the root directory of the project to securely house your API tokens:

Fragmento de código
HF_TOKEN=your_fine_grained_huggingface_token_here
💻 Execution
To run the data ingestion and vector storage pipeline, execute the main entry point:

Bash
python3 main.py
Expected Output Structure
Upon a successful lifecycle run, the console will output execution checkpoints, and a new directory named chroma_db/ will be generated locally:

Plaintext
Generando embeddings y guardando en ChromaDB...
¡Éxito! Se han guardado 4 elementos en la base de datos vectorial.