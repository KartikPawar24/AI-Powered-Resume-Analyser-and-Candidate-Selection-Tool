"""
Configuration settings for data paths, model, embeddings, 
chunking, and retrieval for document processing and search.
"""

from pathlib import Path

# Directories for storing data and vector embeddings
DATA_DIR = Path("data")
VECTORSTORE_PATH = Path("vectorstore")

# Language model configuration
MODEL_CONFIG = {
    "model": "1phi4-mini:3.8b",  # Model identifier
    "temperature": 0,           # Deterministic outputs
    "num_ctx": 2048,            # Context window size
    "num_thread": 6             # Number of threads for inference
}

# Embedding model used for vector representation
EMBEDDING_MODEL = "nomic-embed-text"

# Text chunking configuration for document processing
CHUNK_CONFIG = {
    "chunk_size": 800,           # Max tokens per chunk
    "chunk_overlap": 50,         # Overlap between chunks
    "separators": ["\n\n", "\n", ".", " ", ""]  # Split priorities
}

# Retrieval/search configuration
RETRIEVAL_CONFIG = {
    "search_type": "similarity",  # Search method
    "k": 6                        # Number of top results
}
