"""LangChain tools for the Legal AI agent."""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import SecretStr, BaseModel, Field
from agent_system.utils.Constants import Constants
from agent_system.utils.vector_db import MilvusVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy initialization
_constitution_db: Optional[MilvusVectorDB] = None
_ipc_db: Optional[MilvusVectorDB] = None
_llm: Optional[ChatOpenAI | ChatGoogleGenerativeAI] = None

class KeywordResponse(BaseModel):
    keywords: List[str] = Field(
        description="Array of legal keywords/phrases for search",
        min_length=1
    )

########################################################
#Helper functions
########################################################
def get_constitution_db():
    """Lazy initialization of constitution database."""
    global _constitution_db
    if _constitution_db is None:
        logger.info("🔄 Initializing Constitution database...")
        try:
            _constitution_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_COI if Constants.MILVUS_URI_DB_COI else "",
                token=Constants.MILVUS_TOKEN_DB_COI if Constants.MILVUS_TOKEN_DB_COI else "",
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_CONSTITUTION}_{i}" for i in range(1, Constants.MILVUS_COLLECTION_COUNT_CONSTITUTION + 1)],
            )
            logger.info("✅ Constitution database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Could not initialize constitution database: {e}")
            _constitution_db = None
    return _constitution_db


def get_ipc_db():
    """Lazy initialization of IPC database."""
    global _ipc_db
    if _ipc_db is None:
        logger.info("🔄 Initializing IPC database...")
        try:
            _ipc_db = MilvusVectorDB(
                uri=Constants.MILVUS_URI_DB_IPC if Constants.MILVUS_URI_DB_IPC else "",
                token=Constants.MILVUS_TOKEN_DB_IPC if Constants.MILVUS_TOKEN_DB_IPC else "",
                collection_names=[f"{Constants.MILVUS_COLLECTION_NAME_IPC}_{i}" for i in range(1, Constants.MILVUS_COLLECTION_COUNT_IPC + 1)],
            )
            logger.info("✅ IPC database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Could not initialize IPC database: {e}")
            _ipc_db = None
    return _ipc_db


def get_llm():
    """Lazy initialization of LLM."""
    global _llm
    if _llm is None:
        logger.info("🔄 Initializing LLM...")
        if Constants.LLM_MODEL_NAME == "gpt-4o-mini":
            _llm = ChatOpenAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=SecretStr(Constants.OPENAI_API_KEY) if Constants.OPENAI_API_KEY else None)
        elif Constants.LLM_MODEL_NAME == "gemini-2.0-flash-exp":
            _llm = ChatGoogleGenerativeAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=SecretStr(Constants.GOOGLE_API_KEY) if Constants.GOOGLE_API_KEY else None)
        else:
            raise ValueError(f"Invalid LLM model: {Constants.LLM_MODEL_NAME}")
        
        logger.info("✅ LLM initialized successfully")
    return _llm


########################################################
#Tools
########################################################
@tool
def generate_keywords(query: str) -> str:
    """Generates semantic keywords or phrases for a legal query to improve search results. Can generate single keywords, short phrases, or multiple terms based on query complexity. This tool is designed to handle both single-domain and cross-domain queries effectively."""
    logger.info(f"🔑 TOOL: generate_keywords called with query: '{query}'")
    
    # Initialize structured output parser
    parser = PydanticOutputParser(pydantic_object=KeywordResponse)
    
    prompt = f"""You are a legal search expert.

    **Goal**  
    Turn the user's natural–language legal query into the **smallest set of high-value search terms** (keywords or short phrases) that will maximise semantic-search recall in:

    • Indian Penal Code (IPC) vector DB - populated using text from IPC pdf provided on government website
    • Constitution of India vector DB - populated using text from constitution of india pdf provided on government website

    **What counts as a keyword/phrase**  
    • A single legal term – e.g., "defamation", "trespass"  
    • A short legal phrase – e.g., "forced confession", "office of profit"  
    • A precise citation when clearly relevant – e.g., "Section 302", "Article 19"

    **Rules**  
    1. Return **1 – 6** items. You may generate more keywords for complex cross-domain queries that span both constitutional and criminal law.
    2. Use wording likely found in the IPC or the Constitution (avoid generic fillers like "law", "penalty").  
    3. If a specific Article/Section is obviously implicated, include it exactly once.  
    4. Keep items distinct; no redundant variations.  
    5. For cross-domain queries, generate keywords that will work well in respective databases.
    6. Break the user's query into sub-queries based on the context and generate keywords for each sub-query.

    **Examples**

    User → *"Can freedom of speech be limited in India?"*  
    Keywords: ["freedom of speech", "reasonable restrictions", "Article 19"]

    User → *"What is the punishment for stabbing someone to death?"*  
    Keywords: ["murder", "stabbing", "Section 302", "punishment for homicide"]

    User → *"Protection against arbitrary arrest under Indian Constitution"*  
    Keywords: ["arbitrary arrest", "Article 22", "personal liberty"]

    User → *"Police tortured a suspect to make him confess"*  
    Keywords: ["custodial torture", "forced confession", "Section 330", "police abuse"]

    User → *"Constitutional protections and IPC provisions on hate speech"*  
    Keywords: ["hate speech", "Article 19", "reasonable restrictions", "Section 153A", "promoting enmity", "freedom of expression"]

    ---

    User query: {query}

    {parser.get_format_instructions()}
    """

    try:
        logger.info("🔄 Calling LLM to generate keywords with structured output...")
        llm = get_llm()
        response = llm.invoke(prompt)
        
        # Parse the structured response
        try:
            parsed_response = parser.parse(str(response.content))
            keywords = parsed_response.keywords
            
            # Ensure we have keywords
            if len(keywords) < 1:
                logger.warning(f"⚠️ Only {len(keywords)} keyword(s) generated, should be 1-6")
                keywords = [query]
                
            logger.info(f"📋 Generated {len(keywords)} keywords using structured output: {keywords}")
            return str(keywords)
            
        except Exception as parse_error:
            logger.warning(f"⚠️ Structured parsing failed: {parse_error}, falling back to manual parsing")
            
            # Fallback to manual JSON parsing
            import json
            keywords_response = str(response.content).strip()
            
            try:
                # Try to extract JSON array from the response
                if '[' in keywords_response and ']' in keywords_response:
                    start = keywords_response.find('[')
                    end = keywords_response.rfind(']') + 1
                    json_part = keywords_response[start:end]
                    parsed_keywords = json.loads(json_part)
                    
                    if isinstance(parsed_keywords, list):
                        keywords = parsed_keywords
                        logger.info(f"📋 Fallback parsing successful: {len(keywords)} keywords")
                        return str(keywords)
                
                # If all else fails, treat as single keyword
                clean_response = keywords_response.replace('"', '').replace("'", "").strip()
                keywords = [clean_response]
                logger.info(f"📋 Final fallback: single keyword '{clean_response}'")
                return str(keywords)
                
            except Exception as fallback_error:
                logger.error(f"❌ All parsing methods failed: {fallback_error}")
                keywords = [query]
                return str(keywords)
            
    except Exception as e:
        error_msg = f"Error generating keywords: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return str([error_msg])


@tool
def search_constitution(query: str) -> str:
    """Search the Indian Constitution database using basic search for most relevant articles, clauses, and amendments. 
    
    IMPORTANT: This tool should ONLY be used AFTER calling generate_keywords first. Use individual keywords from the generated keyword list for targeted searches. Returns top 2 most relevant results."""
    logger.info(f"📜 TOOL: search_constitution called with keyword: '{query}'")
    
    constitution_db = get_constitution_db()
    
    if constitution_db is None:
        error_msg = "Constitution database is not available. Please check your database configuration."
        logger.warning(f"⚠️ {error_msg}")
        return error_msg
    
    try:
        logger.info("🔍 Performing basic search on Constitution database...")
        results = constitution_db.search(query, top_k=2)
        
        if not results:
            logger.info("📭 No relevant constitutional provisions found")
            return "No relevant constitutional provisions found for this query."
        
        logger.info(f"📊 Found {len(results)} search results from Constitution database")
        
        formatted_results = []
        for i, result in enumerate(results):
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', 'No content available')
            distance = result.get('distance', 'Unknown')
            collection = result.get('collection', 'Unknown')
            article = entity.get('article', 'Unknown Article')
            
            formatted_results.append(
                f"Result {i+1} (Distance: {distance:.4f}):\n"
                f"Collection: {collection}\n"
                f"Article: {article}\n"
                f"Content: {content}\n"
            )
            logger.debug(f"📄 Result {i+1}: Article {article}, Distance: {distance:.4f}")
        
        result_text = "\n".join(formatted_results)
        logger.info(f"✅ Constitution basic search completed, returning {len(formatted_results)} results")
        return result_text
        
    except Exception as e:
        error_msg = f"Error searching constitution database: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


@tool
def search_ipc(query: str) -> str:
    """Search the Indian Penal Code database using basic search for most relevant sections and offenses. 
    
    IMPORTANT: This tool should ONLY be used AFTER calling generate_keywords first. Use individual keywords from the generated keyword list for targeted searches. Returns top 2 most relevant results."""
    logger.info(f"⚖️ TOOL: search_ipc called with keyword: '{query}'")
    
    ipc_db = get_ipc_db()
    
    if ipc_db is None:
        error_msg = "IPC database is not available. Please check your database configuration."
        logger.warning(f"⚠️ {error_msg}")
        return error_msg
    
    try:
        logger.info("🔍 Performing basic search on IPC database...")
        results = ipc_db.search(query, top_k=2)
        
        if not results:
            logger.info("📭 No relevant IPC sections found")
            return "No relevant IPC sections found for this query."
        
        logger.info(f"📊 Found {len(results)} search results from IPC database")
        
        formatted_results = []
        for i, result in enumerate(results):
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', 'No content available')
            distance = result.get('distance', 'Unknown')
            collection = result.get('collection', 'Unknown')
            section = entity.get('section', 'Unknown Section')
            
            formatted_results.append(
                f"Result {i+1} (Distance: {distance:.4f}):\n"
                f"Collection: {collection}\n"
                f"Section: {section}\n"
                f"Content: {content}\n"
            )
            logger.debug(f"📄 Result {i+1}: Section {section}, Distance: {distance:.4f}")
        
        result_text = "\n".join(formatted_results)
        logger.info(f"✅ IPC basic search completed, returning {len(formatted_results)} results")
        return result_text
        
    except Exception as e:
        error_msg = f"Error searching IPC database: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


@tool
def predict_punishment(case_description: str) -> str:
    """Predict likely punishment and relevant IPC sections based on case description."""
    logger.info(f"🔮 TOOL: predict_punishment called with case: '{case_description[:100]}...'")
    
    prompt = f"""Based on the following case description, predict the likely punishment and relevant IPC sections under Indian law.
    
    Case Description: {case_description}
    
    Please provide:
    1. Likely punishment (imprisonment duration, fine amount, etc.)
    2. Relevant IPC sections
    3. Brief reasoning
    
    Keep the response concise and factual."""
    
    try:
        logger.info("🔄 Calling LLM to predict punishment...")
        llm = get_llm()
        response = llm.invoke(prompt)
        prediction = response.content
        logger.info(f"✅ Punishment prediction completed (length: {len(prediction)} chars)")
        return str(prediction)
    except Exception as e:
        error_msg = f"Error predicting punishment: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


# List of all tools
tools = [generate_keywords, search_constitution, search_ipc, predict_punishment]
logger.info(f"🛠️ Tools module updated with {len(tools)} tools: {[tool.name for tool in tools]}")

