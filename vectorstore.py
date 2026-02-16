"""
Vector store initialization and retriever setup for RAG pipeline.
"""

import logging
from typing import List, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import VECTORSTORE_PATH, EMBEDDING_MODEL, RETRIEVAL_CONFIG


def build_vectorstore(chunks: List[Any]) -> Any:
    """
    Build a vector store from document chunks and return a retriever.

    Args:
        chunks (List[Any]): List of document chunks.

    Returns:
        Any: Retriever instance for document retrieval.
        
    Raises:
        Logs errors but returns None on failure.
    """
    try:
        if not chunks:
            logging.warning("No chunks provided for vector store building")
            return None

        logging.info("Building vector store with %d chunks", len(chunks))

        # Initialize embeddings model
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            logging.info("Embeddings model initialized: %s", EMBEDDING_MODEL)
        except Exception as e:
            logging.error("Failed to initialize embeddings model: %s", str(e))
            raise

        # Create or load vector store
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(VECTORSTORE_PATH)
            )
            logging.info("Vector store created successfully")
        except Exception as e:
            logging.error("Failed to create vector store: %s", str(e))
            raise

        # Create retriever from vector store
        try:
            retriever = vectorstore.as_retriever(
                search_type=RETRIEVAL_CONFIG["search_type"],
                search_kwargs={"k": RETRIEVAL_CONFIG["k"]}
            )
            logging.info("Retriever created with search_type=%s, k=%d", 
                        RETRIEVAL_CONFIG["search_type"], 
                        RETRIEVAL_CONFIG["k"])
        except Exception as e:
            logging.error("Failed to create retriever: %s", str(e))
            raise

        return retriever

    except Exception as e:
        logging.error("Error building vector store: %s", str(e))
        return None
