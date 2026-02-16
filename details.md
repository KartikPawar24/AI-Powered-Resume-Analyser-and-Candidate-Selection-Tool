# AI-Powered Resume Analyzer and Candidate Selction

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/offline-100%25-lightgrey)
![RAG + GenAI Ready](https://img.shields.io/badge/RAG%20%2B%20GenAI-ready-orange)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to evaluate candidate resumes for a project.

Two local LLMs were compared:

- **gemma3:4b**
- **phi4-mini:3.8b**

The experiment was conducted 100% offline using Ollama, ensuring full data privacy.

---

## 📑 Table of Contents
- [AI-Powered Resume Analyzer and Candidate Selction](#ai-powered-resume-analyzer-and-candidate-selction)
  - [📑 Table of Contents](#-table-of-contents)
  - [🎯 Objective](#-objective)
  - [💻 Hardware \& System Configuration](#-hardware--system-configuration)
    - [Why This Matters](#why-this-matters)
  - [🧠 What is RAG?](#-what-is-rag)
    - [Pipeline Used](#pipeline-used)
    - [Architecture & Code Organization](#architecture--code-organization)
  - [💬 Prompt Template Used](#-prompt-template-used)
  - [Why This Prompt Matters](#why-this-prompt-matters)
  - [Query Used](#query-used)
  - [📂 Dataset](#-dataset)
  - [⏱️ Performance Analysis (On Above Hardware)](#️-performance-analysis-on-above-hardware)
  - [📊 Performance Comparison](#-performance-comparison)
  - [🖥️ Hardware Impact Analysis](#️-hardware-impact-analysis)
  - [🤖 Model Behavior Comparison](#-model-behavior-comparison)
  - [🧩 Candidate Outcome](#-candidate-outcome)
    - [🥇 Rohan Sharma](#-rohan-sharma)
    - [🥈 Priya Menon](#-priya-menon)
  - [🔎 Key Experimental Insights](#-key-experimental-insights)
  - [📈 Scalability Considerations](#-scalability-considerations)
  - [🏆 Final Recommendations](#-final-recommendations)
    - [Use `phi4-mini:3.8b` when:](#use-phi4-mini38b-when)
    - [Use `gemma3:4b` when:](#use-gemma34b-when)
  - [🛠 Tech Stack](#-tech-stack)
  - [🔐 Privacy Advantage](#-privacy-advantage)
  - [📌 Conclusion](#-conclusion)


---

## 🎯 Objective

To compare LLM performance for:

- Technical candidate screening
- Multi-role team reasoning
- Complementary skill mapping
- Response time comparison
- Hardware efficiency

## 💻 Hardware & System Configuration

This experiment was executed on:

| Component        | Specification                                                                  |
| ---------------- | ------------------------------------------------------------------------------ |
| **RAM**          | 12 GB                                                                          |
| **Storage**      | SSD                                                                            |
| **Processor**    | Intel® Core™ i5-8250U CPU @ 1.60GHz (up to 1.80 GHz)                           |
| **Architecture** | CPU-only (No GPU acceleration)                                                 |
| **Environment**  | Anaconda base environment + project-specific `langchain-env` using `python -m` |
| **LLM Runtime**  | Ollama                                                                         |

**Note:**

- Python packages were installed inside an Anaconda environment named `langchain-env` using `python -m` for isolation and reproducibility.
- The project is **also fully reproducible without Anaconda** by creating a virtual environment using `python -m venv <env_name>` and installing required packages with `pip install -r requirements.txt`.

### Why This Matters

- No GPU was used.
- All inference was CPU-based.
- Performance reflects real-world mid-range laptop hardware.
- Suitable for small-scale HR automation systems.
- Demonstrates that RAG + 4B parameter models can run effectively on consumer hardware.

## 🧠 What is RAG?

Retrieval-Augmented Generation improves LLM outputs by grounding them in external documents.

### Pipeline Used

1. Load PDF resumes
2. Split into chunks
3. Generate embeddings (Ollama)
4. Store in ChromaDB
5. Retrieve relevant chunks
6. Generate grounded candidate analysis

### Architecture & Code Organization

The project follows a modular, production-ready architecture:

```
├── main.py                    # Entry point - orchestrates the RAG pipeline
├── config.py                  # Centralized configuration
├── llm_factory.py             # LLM & prompt template creation
├── document_loader.py         # PDF loading & text extraction
├── text_processor.py          # Document chunking logic
├── vectorstore.py             # ChromaDB initialization & retrieval setup
├── rag_pipeline.py            # Core RAG execution (query → context → answer)
├── requirements.txt           # Dependencies
└── data/                      # Input: PDF resumes
```

**Module Responsibilities:**

- **config.py**: Single source of truth for all configuration (models, chunk sizes, retrieval params)
- **llm_factory.py**: Instantiates the LLM and creates the prompt template with guardrails
- **document_loader.py**: Uses PyPDFLoader to extract text and preserve metadata
- **text_processor.py**: Applies RecursiveCharacterTextSplitter for context-aware chunking
- **vectorstore.py**: Manages Chroma vector store lifecycle (init, embedding, persistence)
- **rag_pipeline.py**: Simple retrieval-generation loop with context formatting
- **main.py**: Wires all components and provides interactive Q&A interface

This modular design allows easy swapping of components (e.g., different embeddings, different LLMs, or caching strategies).

## 💬 Prompt Template Used

To ensure fairness and reproducibility, the same prompt template was used for both models.

```python
prompt_template = """
You are a senior technical recruiter reviewing one or more resumes.

Answer the question using the information provided in the context.

Guidelines:
- Base your answer strictly on the resume content.
- Do NOT use outside knowledge.
- You may summarize, analyze, and compare candidates when appropriate.
- If some details are missing, answer using the available information without inventing new facts.
- Do NOT fabricate skills, experience, or qualifications that are not mentioned.

Context:
{context}

Question:
{question}

Answer:
"""
```

## Why This Prompt Matters

- Enforces grounded reasoning
- Reduces hallucinations
- Prevents fabrication of skills
- Encourages structured comparison
- Keeps evaluation consistent across models

## Query Used
> "Select a candidate who fits for my GenAI project and can work with LangChain, LLMs, etc."
- **Note:** The output may differ and you may not get the same output as mine.


## 📂 Dataset

| Candidate Role    | Name         |
| ----------------- | ------------ |
| UX/UI Designer    | Ankit Gupta  |
| Data Analyst      | Priya Menon  |
| Software Engineer | Rohan Sharma |
| Project Manager   | Meera Iyer   |

- **Total resumes:** 4
- **Total chunks:** 8
- **Retrieved per query:** 6

## ⏱️ Performance Analysis (On Above Hardware)

Based on execution logs:

| Metric               | gemma3:4b | phi4-mini:3.8b |
| -------------------- | --------- | -------------- |
| Embedding generation | ~3 sec    | ~3 sec         |
| Response generation  | ~120 sec  | ~95 sec        |
| Total response time  | ~125 sec  | ~100 sec       |

## 📊 Performance Comparison

| Metric                  | gemma3:4b | phi4-mini:3.8b |
| ----------------------- | --------- | -------------- |
| Avg Response Time       | ~125 sec  | ~100 sec       |
| Relative Speed          | Slower    | ~20–25% Faster |
| CPU Load                | High      | Moderate       |
| RAM Usage               | Higher    | Slightly Lower |
| Suitability on 12GB RAM | Stable    | Very Stable    |

## 🖥️ Hardware Impact Analysis

**System Used:**

- 12GB RAM
- CPU-only (no GPU)
- 8th Gen i5 processor

**Observations:**

- 4B parameter models are viable on mid-range laptops.
- gemma3 shows longer generation time due to deeper reasoning.
- phi4-mini is more efficient under CPU constraints.
- SSD helps reduce vector store load times.
- GPU acceleration would significantly reduce response time.

## 🤖 Model Behavior Comparison

| Aspect          | gemma3:4b | phi4-mini:3.8b |
| --------------- | --------- | -------------- |
| Top Candidate   | Rohan     | Rohan          |
| Secondary       | Priya     | Priya          |
| Reasoning Style | Holistic  | Structured     |
| Verbosity       | Moderate  | High           |
| Team Awareness  | Strong    | Moderate       |
| Speed           | Slower    | Faster         |

## 🧩 Candidate Outcome

### 🥇 Rohan Sharma

- Python, Django, React
- Microservices & APIs
- Docker & AWS
- Suitable for LLM integration & deployment

### 🥈 Priya Menon

- Python (Pandas, NumPy)
- SQL
- Data processing & analytics
- Strong support for GenAI data workflows

## 🔎 Key Experimental Insights

- Both models correctly prioritized technical candidates.
- gemma3 is better for team-level reasoning.
- phi4-mini is better for faster screening.
- On 12GB RAM, both models run reliably.
- CPU-only inference increases response time but remains usable.
- Prompt constraints helped reduce hallucination risk.
- RAG grounding improved factual reliability.

## 📈 Scalability Considerations

On similar hardware:

- Increasing resume count → retrieval time increases slightly
- Increasing chunk size → improves context but increases latency
- Using quantized models → improves speed
- Reducing retrieved chunks → faster generation

For production-level systems:

- Add GPU
- Implement response streaming
- Add embedding cache
- Add structured JSON outputs

## 🏆 Final Recommendations

### Use `phi4-mini:3.8b` when:

- Running on limited hardware
- Prioritizing speed
- Doing bulk resume screening

### Use `gemma3:4b` when:

- Planning team composition
- Needing richer reasoning
- Providing management-level insights

## 🛠 Tech Stack

### Core Framework
- **LangChain** — LLM orchestration and RAG framework
  - `langchain==1.2.9`
  - `langchain-core==1.2.9` — core abstractions
  - `langchain-community==0.4.1` — document loaders & integrations
  - `langchain-ollama==1.0.1` — Ollama LLM & embeddings
  - `langchain-chroma==1.1.0` — ChromaDB integration
  - `langchain-text-splitters==1.1.0` — recursive text chunking

### LLM & Embeddings Runtime
- **Ollama** — local LLM runtime
  - `ollama==0.6.1` — Python client
- **Supported Models**:
  - `phi4-mini:3.8b` — primary model (faster, 3.8B params)
  - `gemma3:4b` — alternative comparison model (4B params)
  - `nomic-embed-text` — embedding model

### Vector Store & Retrieval
- **ChromaDB** — vector database
  - `chromadb==1.5.0`
  - OpenTelemetry for observability
  - SQLAlchemy for database operations

### Document Processing
- **PDF Parsing**: `pypdf==6.7.0` — extracts text from PDFs
- **Text Splitting**: Recursive character-level splitting with configurable overlaps
- **Chunking Strategy**: 800-character chunks with 50-character overlap

### Data & Validation
- `pydantic==2.12.5` — schema validation
- `pydantic-settings==2.12.0` — configuration management
- `python-dotenv==1.2.1` — environment variable loading

### Network & Async
- `httpx==0.28.1` — async HTTP client
- `aiohttp==3.13.3` — async HTTP library
- `anyio==4.12.1` — async backend abstraction
- `websockets==16.0` — WebSocket support

### Utilities
- `click==8.3.1` — CLI framework
- `rich==14.3.2` — rich terminal output
- `python-dateutil==2.9.0.post0` — date utilities
- `PyYAML==6.0.3` — YAML parsing
- `tqdm==4.67.3` — progress bars

### Development & Debugging
- `ipython==9.10.0` — interactive shell
- `jupyter_client==8.8.0` — Jupyter support
- `debugpy==1.8.20` — debugging support

For complete dependency list, see [requirements.txt](requirements.txt).

## 🔐 Privacy Advantage

- 100% Offline
- No external API calls
- Resume data never leaves machine
- Suitable for HR-sensitive environments

## 📌 Conclusion

This experiment proves that:

- RAG + 4B parameter LLMs can run on mid-range laptops (12GB RAM).
- Model choice impacts speed, reasoning depth, and hardware load.
- Prompt engineering plays a critical role in grounding responses.
- Offline LLM systems can provide actionable hiring insights.

This project demonstrates a practical, private, and scalable candidate screening system for projects.

[⬅ Back to README](README.md)