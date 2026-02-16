"""
LLM and prompt configuration for resume evaluation.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_CONFIG

# Prompt template used for resume-based question answering
PROMPT_TEMPLATE = """
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


def create_llm() -> ChatOllama:
    """
    Create and return a configured ChatOllama language model instance.
    
    Returns:
        ChatOllama: Configured language model instance.
        
    Raises:
        Exception: If LLM initialization fails.
    """
    try:
        llm = ChatOllama(**MODEL_CONFIG)
        logging.info("LLM initialized successfully with model: %s", MODEL_CONFIG.get("model"))
        return llm
    except Exception as e:
        logging.error("Failed to initialize LLM: %s", str(e))
        raise


def create_prompt() -> ChatPromptTemplate:
    """
    Create and return the chat prompt template for resume evaluation.
    
    Returns:
        ChatPromptTemplate: Chat prompt template.
        
    Raises:
        Exception: If prompt template creation fails.
    """
    try:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        logging.info("Prompt template created successfully")
        return prompt
    except Exception as e:
        logging.error("Failed to create prompt template: %s", str(e))
        raise
