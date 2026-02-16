# AI-Powered Resume Analyser and Candidate Selction

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/offline-100%25-lightgrey)
![RAG + GenAI Ready](https://img.shields.io/badge/RAG%20%2B%20GenAI-ready-orange)

## Summary

AI-Powered Resume Analyser and Candidate Selction is a lightweight, local-first resume ingestion and question-answering tool built with LangChain components, Ollama embeddings/LLMs and Chroma vector store. It converts PDF resumes into searchable vectors and provides an interactive prompt to query candidate details (skills, experience, education, etc.).

## Screenshot:
![Screenshot-1](./screenshots/screenshot_1.png)

For deeper documentation and implementation notes, see [here](details.md). 

## Key Features

- PDF resume ingestion and text extraction
- Document chunking for better retrieval
- Embedding generation via Ollama (`nomic-embed-text`) and vector storage with Chroma
- Chat-style question-answering using a local Ollama LLM
- Fast, minimal dependency surface — designed for local/private usage

## Project Layout

- `main.py` — primary script: orchestrates the RAG pipeline (loads PDFs, creates vectorstore, starts interactive QA session)
- `config.py` — centralized configuration (model params, chunk settings, retrieval config)
- `llm_factory.py` — creates LLM and prompt template for the RAG pipeline
- `document_loader.py` — loads and extracts text from PDF resumes
- `text_processor.py` — chunks documents using `RecursiveCharacterTextSplitter`
- `vectorstore.py` — builds and persists Chroma vectorstore with Ollama embeddings
- `rag_pipeline.py` — core RAG logic (retrieval + generation)
- `data/` — place PDF resumes here (project expects `*.pdf`)
- `vectorstore/` — created by the app to persist embeddings (safe to delete and rebuild)

## Requirements

- Python 3.11+ (virtual environment recommended)
- Ollama installed and running locally (for embeddings & chat models)
- Access to Ollama models:
  - `nomic-embed-text` — for generating embeddings
  - `phi4-mini:3.8b` — for candidate analysis (default; `gemma3:4b` is also supported)

### Core Dependencies

All Python packages are captured in `requirements.txt`. Key packages include:

- **LangChain Ecosystem**: 
  - `langchain-core==1.2.9` — core abstractions
  - `langchain-community==0.4.1` — integrations (PyPDFLoader)
  - `langchain-ollama==1.0.1` — Ollama LLM & embedding support
  - `langchain-chroma==1.1.0` — ChromaDB vector store
  - `langchain-text-splitters==1.1.0` — document chunking
  
- **Vector Store & Embeddings**:
  - `chromadb==1.5.0` — vector database
  
- **PDF Processing**:
  - `pypdf==6.7.0` — PDF text extraction
  
- **LLM Integration**:
  - `ollama==0.6.1` — Ollama client library
  - `pydantic==2.12.5` — data validation

See `requirements.txt` for the complete dependency list.

## Setup (Quick)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: .\.venv\Scripts\activate.bat  # cmd.exe
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure Ollama is installed and the models referenced in `main.py` are available locally. For embeddings the code uses `nomic-embed-text`; for chat it uses `phi4-mini:3.8b` by default. Start Ollama if required.

4. Add one or more PDF resumes to the `data/` folder.

## Run

From the project root (where `main.py` lives):

```powershell
python main.py
```

The script will:

- Read all `*.pdf` files from `data/`
- Chunk documents and create embeddings with Ollama
- Persist a Chroma vectorstore to `vectorstore/`
- Open an interactive prompt where you can ask questions about the resumes (type `exit` to quit)

Example query:

```
What are the key skills listed by Anikit ?
```

## Configuration & Notes

All configuration is centralized in `config.py`:

- **Model**: Default is `phi4-mini:3.8b`; change `MODEL_CONFIG["model"]` to use alternatives (e.g., `gemma3:4b`)
- **Temperature**: Set to `0` for deterministic responses; adjust in `MODEL_CONFIG["temperature"]`
- **Embeddings**: Uses `nomic-embed-text` via Ollama; configured in `EMBEDDING_MODEL`
- **Chunking**: Documents are split into 800-character chunks with 50-character overlap (configurable in `CHUNK_CONFIG`)
- **Retrieval**: By default, retrieves 6 most similar chunks per query (configurable in `RETRIEVAL_CONFIG["k"]`)
- **Prompt**: Tuned to act as a senior technical recruiter; modify `PROMPT_TEMPLATE` in `llm_factory.py` to change behavior

**Important**: The vectorstore is deleted and recreated on each run. Modify `vectorstore.py` if you want to preserve incremental updates.

## Troubleshooting

- If PDFs fail to load, verify they are valid PDF files and readable by the process.
- If Ollama embedding/LLM calls fail, ensure Ollama daemon is running and the named models are installed.
- Permission errors when creating `vectorstore/` — check file permissions and run from a directory where you can create files.

## Security & Privacy

This project runs locally and uses local models/embeddings (Ollama + Chroma). Do not place sensitive resumes in shared or public folders if you cannot guarantee access controls.

## Contributing

PRs welcome. For changes that alter ingestion, embedding models, or storage layout, include notes in the PR explaining backwards compatibility.

## License & Contact

This repository include a license by MIT.

For questions or to open an issue, contact the project maintainer.
 