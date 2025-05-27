import os
class Constants:
    ##################################################################
    # Env Variables
    ##################################################################
    # Milvus
    MILVUS_URI_DB_COI = os.getenv("MILVUS_URI_DB_COI")
    MILVUS_TOKEN_DB_COI = os.getenv("MILVUS_TOKEN_DB_COI")
    MILVUS_URI_DB_IPC = os.getenv("MILVUS_URI_DB_IPC")
    MILVUS_TOKEN_DB_IPC = os.getenv("MILVUS_TOKEN_DB_IPC")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # HuggingFace
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    
    
    
    
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
    2. search_constitution: Search the Indian Constitution database
    3. search_ipc: Search the Indian Penal Code database  
    4. predict_punishment: Predict likely punishment for case descriptions

    Use these tools to provide comprehensive, accurate legal information. Always:
    - Search both Constitution and IPC when relevant
    - Generate keywords first for better search results
    - Cite specific articles/sections in your responses
    - Provide clear, structured answers
    - Use predict_punishment tool when asked about potential penalties

    Answer queries about Indian law with precision and cite your sources.
    """
