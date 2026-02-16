import logging
import sys
from pathlib import Path
from config import DATA_DIR
from llm_factory import create_llm, create_prompt
from document_loader import load_pdfs
from text_processor import chunk_documents
from vectorstore import build_vectorstore
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    """
    Main entry point for the Resume Analyzer RAG application.
    """
    try:
        logging.info("Starting Resume Analyzer application")
        
        # Validate data directory
        if not DATA_DIR.exists():
            logging.error("Data directory does not exist: %s", DATA_DIR)
            print(f"Error: Data directory not found at {DATA_DIR}")
            sys.exit(1)
        
        # Load documents
        try:
            logging.info("Loading PDF documents from %s", DATA_DIR)
            docs = load_pdfs(DATA_DIR)
            
            if not docs:
                logging.warning("No documents found in the data directory")
                print("\nNo documents found in the data directory.")
                print(f"Please add PDF files to: {DATA_DIR}")
                sys.exit(1)
                
            logging.info("Loaded %d documents", len(docs))
            
        except Exception as e:
            logging.error("Failed to load PDF documents: %s", str(e))
            print(f"\nError loading documents: {str(e)}")
            sys.exit(1)

        # Chunk documents
        try:
            logging.info("Chunking documents")
            chunks = chunk_documents(docs)
            
            if not chunks:
                logging.error("No chunks created from documents")
                print("\nError: Failed to process documents into chunks.")
                sys.exit(1)
                
            logging.info("Created %d chunks", len(chunks))
            
        except Exception as e:
            logging.error("Failed to chunk documents: %s", str(e))
            print(f"\nError processing documents: {str(e)}")
            sys.exit(1)

        # Build vectorstore
        try:
            logging.info("Building vectorstore")
            retriever = build_vectorstore(chunks)
            
            if retriever is None:
                logging.error("Failed to create retriever")
                print("\nError: Failed to build vector store for retrieval.")
                sys.exit(1)
                
            logging.info("Vectorstore created successfully")
            
        except Exception as e:
            logging.error("Failed to build vectorstore: %s", str(e))
            print(f"\nError building vector store: {str(e)}")
            sys.exit(1)

        # Initialize LLM and prompt
        try:
            logging.info("Initializing LLM and prompt template")
            llm = create_llm()
            prompt = create_prompt()
            logging.info("LLM and prompt initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize LLM or prompt: %s", str(e))
            print(f"\nError initializing language model: {str(e)}")
            sys.exit(1)

        # Create RAG pipeline
        try:
            logging.info("Creating RAG pipeline")
            rag = RAGPipeline(llm, prompt, retriever)
            logging.info("RAG pipeline ready")
            
        except Exception as e:
            logging.error("Failed to create RAG pipeline: %s", str(e))
            print(f"\nError creating RAG pipeline: {str(e)}")
            sys.exit(1)

        # Interactive question-answering loop
        print("\n" + "="*60)
        print("Resume Analyzer Ready!")
        print("="*60)
        print("\nAsk questions about the resumes.")
        print("Enter 'exit' or 'quit' to end.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    logging.info("User exited the application")
                    break

                answer = rag.ask(question)
                print(f"\nAI: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                logging.info("Application interrupted by user")
                break
            except Exception as e:
                logging.error("Error processing question: %s", str(e))
                print(f"\nError processing your question: {str(e)}")
                print("Please try again.\n")
                
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
        print("\n\nApplication terminated.")
        sys.exit(0)
    except Exception as e:
        logging.critical("Critical error in main: %s", str(e))
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)
    finally:
        logging.info("Resume Analyzer application ended")

if __name__ == "__main__":
    main()
