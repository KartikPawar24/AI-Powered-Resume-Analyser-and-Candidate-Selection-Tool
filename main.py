import logging
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------
# Logging Configuration
# ------------------------------
# Set up logging to display timestamp, log level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ------------------------------
# Paths
# ------------------------------
# Folder where resumes are stored
data_dir = Path("data")
# Folder to persist the vectorstore
vectorstore_path = Path("vectorstore")

# ------------------------------
# Initialize LLM
# ------------------------------
# Using Ollama model (phi4-mini gives more  than gemma3)
llm = ChatOllama(
    model="phi4-mini:3.8b",
    temperature=0,    # deterministic output
    num_ctx=2048,     # context window
    num_thread=6      # number of threads for inference
)

# ------------------------------
# Strict Prompt Template
# ------------------------------
# This ensures the model only uses the provided resume data
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

prompt = ChatPromptTemplate.from_template(prompt_template)

# ------------------------------
# Load PDFs
# ------------------------------
def load_pdf(folder: Path):
    """
    Load all PDF files from the specified folder and return a list of documents.
    Adds the filename as metadata to each document.
    """
    logging.info(f"Scanning folder: {folder}")
    documents = []

    if not folder.exists():
        logging.warning("Data folder does not exist.")
        return documents

    for file in folder.glob("*.pdf"):
        logging.info(f"Loading {file.name}")
        loader = PyPDFLoader(str(file))
        docs = loader.load()

        # Add source metadata for reference
        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

    logging.info(f"Total documents loaded: {len(documents)}")
    return documents

# ------------------------------
# Chunk Documents
# ------------------------------
def documents_chunking(documents):
    """
    Split documents into smaller chunks for better embedding and retrieval.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # target chunk size
        chunk_overlap=50,         # overlap between chunks
        separators=["\n\n", "\n", ".", " ", ""],  # preferred separators
        keep_separator=True
    )

    chunks = splitter.split_documents(documents)
    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

# ------------------------------
# Vectorstore Creation
# ------------------------------
def vectorize_data(chunks):
    """
    Create embeddings for document chunks and store them in Chroma vectorstore.
    Always rebuilds vectorstore to prevent stale data during development.
    """
    if not chunks:
        return None

    # Remove old vectorstore if it exists
    if vectorstore_path.exists():
        logging.info("Deleting old vectorstore...")
        shutil.rmtree(vectorstore_path)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    logging.info("Creating new vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vectorstore_path)
    )

    # Create a retriever to fetch relevant chunks based on similarity
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # number of chunks to retrieve
    )

    logging.info("Retriever ready.")
    return retriever

# ------------------------------
# Ask a Question
# ------------------------------
def ask_question(question, retriever):
    """
    Retrieve relevant document chunks and query the LLM using the strict prompt.
    """
    if not retriever:
        return "Retriever not available."

    # Retrieve the top relevant chunks
    relevant_docs = retriever.invoke(question)
    logging.info(f"Retrieved {len(relevant_docs)} chunks")

    # Combine the chunks into a single context with source info
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_docs
        ]
    )

    # Fill prompt template with context and user question
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    # Handle response content
    answer = response.content if hasattr(response, "content") else str(response)
    return answer

# ------------------------------
# Main Program
# ------------------------------
def main():
    docs = load_pdf(data_dir)
    if not docs:
        logging.warning("No documents found. Exiting.")
        return

    chunks = documents_chunking(docs)
    if not chunks:
        logging.warning("No chunks created. Exiting.")
        return

    retriever = vectorize_data(chunks)
    if not retriever:
        logging.warning("Retriever failed. Exiting.")
        return

    # Interactive loop for asking questions
    while True:
        question = input("\nYou (Enter your question (or 'exit')): ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = ask_question(question, retriever)
        print(f"\nAI: {answer}")

if __name__ == "__main__":
    main()
