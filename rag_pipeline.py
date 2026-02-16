import logging
from typing import Any


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.

    Combines a retriever, prompt template, and language model
    to answer questions based on retrieved context.
    """

    def __init__(self, llm: Any, prompt: Any, retriever: Any) -> None:
        """
        Initialize the RAG pipeline components.

        Args:
            llm: Language model instance.
            prompt: Prompt template instance.
            retriever: Retriever instance for fetching relevant documents.
            
        Raises:
            ValueError: If any component is None.
        """
        if llm is None:
            raise ValueError("LLM instance cannot be None")
        if prompt is None:
            raise ValueError("Prompt instance cannot be None")
        if retriever is None:
            raise ValueError("Retriever instance cannot be None")
            
        self.llm = llm
        self.prompt = prompt
        self.retriever = retriever
        logging.info("RAG Pipeline initialized successfully")

    def ask(self, question: str) -> str:
        """
        Retrieve relevant documents and generate an answer.

        Args:
            question (str): User question.

        Returns:
            str: Model-generated answer or error message.
        """
        try:
            if not question or not question.strip():
                logging.warning("Empty question provided")
                return "Please provide a valid question."
            
            # Retrieve relevant documents
            try:
                logging.info("Retrieving documents for question: %s", question[:50])
                relevant_docs = self.retriever.invoke(question)
                
                if not relevant_docs:
                    logging.warning("No relevant documents retrieved")
                    return "No relevant information found in the documents."
                    
                logging.info("Retrieved %d relevant documents", len(relevant_docs))
                
            except Exception as e:
                logging.error("Error during document retrieval: %s", str(e))
                return "Error retrieving relevant documents. Please try again."

            # Build context string from retrieved documents
            try:
                context = "\n\n".join(
                    f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                    for doc in relevant_docs
                )
                logging.info("Context prepared (length: %d characters)", len(context))
            except Exception as e:
                logging.error("Error preparing context: %s", str(e))
                return "Error processing retrieved documents."

            # Format final prompt
            try:
                final_prompt = self.prompt.format(
                    context=context,
                    question=question
                )
            except Exception as e:
                logging.error("Error formatting prompt: %s", str(e))
                return "Error formatting prompt."

            # Generate response from LLM
            try:
                logging.info("Invoking LLM for answer generation")
                response = self.llm.invoke(final_prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                logging.info("Answer generated successfully")
                return answer
                
            except Exception as e:
                logging.error("Error generating answer from LLM: %s", str(e))
                return "Error generating answer. Please try again."
                
        except Exception as e:
            logging.error("Unexpected error in ask method: %s", str(e))
            return "An unexpected error occurred. Please try again."
