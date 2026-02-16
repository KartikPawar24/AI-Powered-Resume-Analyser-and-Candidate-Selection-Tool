import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import List

def load_pdfs(folder: Path) -> List:
    """
    Load all PDF files from the given folder and return them as a list of documents.
    Each document's metadata is updated with its source file name.
    
    Args:
        folder (Path): Path to the folder containing PDF files.
    
    Returns:
        List: A list of loaded documents.
    
    Raises:
        Logs warnings and errors but returns empty list on failure.
    """
    documents = []

    try:
        # Check if folder exists
        if not folder.exists():
            logging.warning("Data folder does not exist: %s", folder)
            return documents

        # Iterate over all PDFs in the folder
        for file in folder.glob("*.pdf"):
            try:
                logging.info("Loading PDF: %s", file.name)
                loader = PyPDFLoader(str(file))
                docs = loader.load()

                if not docs:
                    logging.warning("No documents extracted from PDF: %s", file.name)
                    continue

                # Add source metadata to each document
                for doc in docs:
                    doc.metadata["source"] = file.name

                documents.extend(docs)
                logging.info("Successfully loaded %d documents from %s", len(docs), file.name)

            except Exception as e:
                logging.error("Failed to load PDF %s: %s", file.name, str(e))
                continue

    except Exception as e:
        logging.error("Error scanning PDF folder: %s", str(e))

    logging.info("Total documents loaded: %d", len(documents))
    return documents
