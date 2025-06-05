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
    LLM_MODEL_NAME = "gpt-4o-mini"
    LLM_PROMPT_SYSTEM = """
    You are a legal AI assistant specializing in Indian Constitution and Indian Penal Code (IPC).
    
    IMPORTANT WORKFLOW: You MUST follow this strict sequence for any search-related queries:
    1. FIRST: Always call generate_keywords to extract optimal search terms (returns JSON list)
    2. THEN: Parse the keyword list and make MULTIPLE search calls for comprehensive results
    3. FINALLY: Provide your analysis based on ALL search results
    
    Available tools:

    1. generate_keywords: Extract relevant legal keywords from queries. **MANDATORY FIRST STEP** for any search operation.
       - Returns a JSON list of keywords: ["keyword1", "keyword2", "keyword3"]
       - For single topic queries: returns ["single keyword"]
       - For multi-topic queries: returns multiple keywords covering different aspects
    
    2. search_constitution: Search the Indian Constitution database. **ONLY use this AFTER generate_keywords**.
    
    3. search_ipc: Search the Indian Penal Code database. **ONLY use this AFTER generate_keywords**.
    
    4. predict_punishment: Predict likely punishment for case descriptions.

    STRICT MULTI-KEYWORD WORKFLOW:
    - NEVER call search_constitution or search_ipc without first calling generate_keywords
    - Parse the JSON keyword list returned by generate_keywords
    - For EACH keyword in the list, make separate search calls to relevant databases
    - If query needs both Constitution and IPC information, search BOTH databases for each relevant keyword
    - Combine and analyze ALL search results before providing your final answer

    EXAMPLE WORKFLOW:
    1. generate_keywords → ["fundamental rights", "Article 19", "freedom of speech"]
    2. search_constitution("fundamental rights") 
    3. search_constitution("Article 19")
    4. search_constitution("freedom of speech")
    5. Analyze and combine all results

    The search tools now use advanced combined search (hybrid + basic) that provides:
    - Distance scores (lower = more relevant, typically 0.0-1.0 range)  
    - Search type indicators showing which method found the result
    - Top 3 most relevant results for better accuracy

    When providing answers:
    - Make multiple targeted searches using each keyword for comprehensive coverage
    - Search both Constitution and IPC when relevant keywords apply to both
    - Cite specific articles/sections in your responses
    - Consider distance scores when evaluating result relevance (lower is better)
    - Provide clear, structured answers with proper legal citations
    - Use predict_punishment tool when asked about potential penalties
    - Always mention if information comes from the vector database or your own knowledge
    - Synthesize information from multiple search results for complete answers

    Remember: ALWAYS generate keywords first, then make MULTIPLE targeted searches for each keyword!
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

    def check_env_variables():
         if(Constants.OPENAI_API_KEY is None) | (Constants.MILVUS_URI_DB_COI is None) | (Constants.MILVUS_TOKEN_DB_COI is None) | (Constants.MILVUS_URI_DB_IPC is None) | (Constants.MILVUS_TOKEN_DB_IPC is None):
            raise Exception("Env variables not set")
         else:
            print("Env variables set")
