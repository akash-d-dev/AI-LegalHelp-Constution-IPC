import os
from dotenv import load_dotenv
class Constants:
    ##################################################################
    # Env Variables
    ##################################################################
    # Milvus
    MILVUS_URI_DB_COI = None
    MILVUS_TOKEN_DB_COI = None
    MILVUS_URI_DB_IPC = None
    MILVUS_TOKEN_DB_IPC = None
    
    # OpenAI
    OPENAI_API_KEY = None
    
    # HuggingFace
    HUGGINGFACE_API_TOKEN = None
    
    
    
    
    ##################################################################
    # Milvis Collection Names
    ##################################################################
    MILVUS_COLLECTION_NAME_CONSTITUTION = "constitution_of_india"
    MILVUS_COLLECTION_COUNT_CONSTITUTION = 5
    MILVUS_COLLECTION_NAME_IPC = "ipc"
    MILVUS_COLLECTION_COUNT_IPC = 2
    
    ##################################################################
    # Embedding Model
    ##################################################################
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    ##################################################################
    # LLM Model
    ##################################################################
    LLM_MODEL_NAME = "gpt-4o"
    LLM_PROMPT_SYSTEM = """
    You are a legal AI assistant specializing in Indian Constitution and Indian Penal Code (IPC).
    You have access to the following tools:

    1. generate_keywords: Extract relevant legal keywords from queries
    2. search_constitution: Search the Indian Constitution database using combined hybrid and basic search
    3. search_ipc: Search the Indian Penal Code database using combined hybrid and basic search
    4. predict_punishment: Predict likely punishment for case descriptions

    The search tools now use advanced combined search (hybrid + basic) that provides:
    - Distance scores (lower = more relevant, typically 0.0-1.0 range)
    - Search type indicators (hybrid/basic) showing which method found the result
    - Top 3 most relevant results for better accuracy

    Use these tools to provide comprehensive, accurate legal information. Always:
    - Search both Constitution and IPC when relevant
    - Generate keywords first for better search results
    - Cite specific articles/sections in your responses
    - Consider distance scores when evaluating result relevance (lower is better)
    - Provide clear, structured answers with proper legal citations
    - Use predict_punishment tool when asked about potential penalties

    When presenting search results, prioritize those with lower distance scores as they are more relevant.
    Answer queries about Indian law with precision and cite your sources.
    """

    def set_env_variables():
        load_dotenv()
        print("Setting env variables")
        Constants.MILVUS_URI_DB_COI = os.getenv("MILVUS_URI_DB_COI")
        Constants.MILVUS_TOKEN_DB_COI = os.getenv("MILVUS_TOKEN_DB_COI")
        Constants.MILVUS_URI_DB_IPC = os.getenv("MILVUS_URI_DB_IPC")
        Constants.MILVUS_TOKEN_DB_IPC = os.getenv("MILVUS_TOKEN_DB_IPC")
        Constants.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        Constants.HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if(Constants.OPENAI_API_KEY is None) | (Constants.MILVUS_URI_DB_COI is None) | (Constants.MILVUS_TOKEN_DB_COI is None) | (Constants.MILVUS_URI_DB_IPC is None) | (Constants.MILVUS_TOKEN_DB_IPC is None):
            raise Exception("Env variables not set")
