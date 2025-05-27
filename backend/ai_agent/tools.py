"""LangChain tools for the Legal AI agent."""

from __future__ import annotations

from typing import List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from utils.Constants import Constants

# Global variables for lazy initialization
_constitution_db: Optional[object] = None
_ipc_db: Optional[object] = None
_llm: Optional[ChatOpenAI] = None


def get_constitution_db():
    """Lazy initialization of constitution database."""
    global _constitution_db
    if _constitution_db is None:
        try:
            from utils.vector_db import MilvusVectorDB
            _constitution_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_COI,
                token=Constants.MILVUS_TOKEN_DB_COI,
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_CONSTITUTION}_{i}" for i in range(1, Constants.MILVUS_COLLECTION_COUNT_CONSTITUTION + 1)],
            )
        except Exception as e:
            print(f"Warning: Could not initialize constitution database: {e}")
            _constitution_db = None
    return _constitution_db


def get_ipc_db():
    """Lazy initialization of IPC database."""
    global _ipc_db
    if _ipc_db is None:
        try:
            from utils.vector_db import MilvusVectorDB
            _ipc_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_IPC,
                token=Constants.MILVUS_TOKEN_DB_IPC,
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_IPC}_{i}" for i in range(1, Constants.MILVUS_COLLECTION_COUNT_IPC + 1)],
            )
        except Exception as e:
            print(f"Warning: Could not initialize IPC database: {e}")
            _ipc_db = None
    return _ipc_db


def get_llm():
    """Lazy initialization of LLM."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(temperature=0, api_key=Constants.OPENAI_API_KEY)
    return _llm


@tool
def generate_keywords(query: str) -> str:
    """Generate semantic keywords for a legal query to improve search results."""
    prompt = f"""Extract important legal keywords from this query for better search results.
    Return only the keywords separated by commas, no explanations.
    
    Query: {query}
    
    Keywords:"""
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error generating keywords: {str(e)}"


@tool
def search_constitution(query: str) -> str:
    """Search the Indian Constitution database for relevant articles, clauses, and amendments."""
    constitution_db = get_constitution_db()
    
    if constitution_db is None:
        return "Constitution database is not available. Please check your database configuration."
    
    try:
        results = constitution_db.search(query)
        if not results:
            return "No relevant constitutional provisions found for this query."
        
        formatted_results = []
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            content = result.get('text', result.get('content', 'No content available'))
            metadata = result.get('metadata', {})
            article = metadata.get('article', 'Unknown Article')
            
            formatted_results.append(f"Result {i+1}:\nArticle: {article}\nContent: {content}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching constitution database: {str(e)}"


@tool
def search_ipc(query: str) -> str:
    """Search the Indian Penal Code database for relevant sections and offenses."""
    ipc_db = get_ipc_db()
    
    if ipc_db is None:
        return "IPC database is not available. Please check your database configuration."
    
    try:
        results = ipc_db.search(query)
        if not results:
            return "No relevant IPC sections found for this query."
        
        formatted_results = []
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            content = result.get('text', result.get('content', 'No content available'))
            metadata = result.get('metadata', {})
            section = metadata.get('section', 'Unknown Section')
            
            formatted_results.append(f"Result {i+1}:\nSection: {section}\nContent: {content}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching IPC database: {str(e)}"


@tool
def predict_punishment(case_description: str) -> str:
    """Predict likely punishment and relevant IPC sections based on case description."""
    prompt = f"""Based on the following case description, predict the likely punishment and relevant IPC sections under Indian law.
    
    Case Description: {case_description}
    
    Please provide:
    1. Likely punishment (imprisonment duration, fine amount, etc.)
    2. Relevant IPC sections
    3. Brief reasoning
    
    Keep the response concise and factual."""
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error predicting punishment: {str(e)}"


# List of all tools for easy import
tools = [generate_keywords, search_constitution, search_ipc, predict_punishment]

