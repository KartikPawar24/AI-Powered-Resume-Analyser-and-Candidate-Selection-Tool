"""
Document chunking utilities.
"""

import logging
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_CONFIG


def chunk_documents(documents: List[Any]) -> List[Any]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Args:
        documents (List[Any]): List of loaded documents.

    Returns:
        List[Any]: List of chunked documents. Empty list if operation fails.
        
    Raises:
        Logs errors but returns empty list on failure.
    """
    try:
        # Return early if no documents are provided
        if not documents:
            logging.warning("No documents provided for chunking")
            return []

        logging.info("Starting document chunking with %d documents", len(documents))

        # Initialize text splitter with configured parameters
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
            separators=CHUNK_CONFIG["separators"],
            keep_separator=True,
        )

        chunks = splitter.split_documents(documents)
        logging.info("Successfully created %d chunks", len(chunks))
        
        if not chunks:
            logging.warning("No chunks created from documents")
            
        return chunks
        
    except Exception as e:
        logging.error("Error during document chunking: %s", str(e))
        return []
