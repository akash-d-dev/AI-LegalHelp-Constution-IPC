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
    """Generates semantic keywords or phrases for a legal query to improve search results. Can generate single keywords, short phrases, or multiple terms based on query complexity."""
    logger.info(f"🔑 TOOL: generate_keywords called with query: '{query}'")
    
    # Initialize structured output parser
    parser = PydanticOutputParser(pydantic_object=KeywordResponse)
    
    prompt = f"""You are a legal search expert.

    **Goal**  
    Turn the user's natural–language legal query into the **smallest set of high-value search terms** (keywords or short phrases) that will maximise semantic-search recall in:

    • Indian Penal Code (IPC) vector DB  
    • Constitution of India vector DB  

    **What counts as a keyword/phrase**  
    • A single legal term – e.g., "defamation", "trespass"  
    • A short legal phrase – e.g., "forced confession", "office of profit"  
    • A precise citation when clearly relevant – e.g., "Section 302", "Article 19"

    **Rules**  
    1. Return **1 – 4** items. You *may* exceed 4 **only** if the query is complex and extra terms will clearly improve recall.  
    2. Use wording likely found in the IPC or the Constitution (avoid generic fillers like "law", "penalty").  
    3. If a specific Article/Section is obviously implicated, include it exactly once.  
    4. Keep items distinct; no redundant variations.  
    5. Break the user's query into sub-queries based on the context and generate keywords for each sub-query.

    **Examples**

    User → *"Can freedom of speech be limited in India?"*  
    Keywords: ["freedom of speech", "reasonable restrictions", "Article 19"]

    User → *"What is the punishment for stabbing someone to death?"*  
    Keywords: ["murder", "stabbing", "Section 302", "punishment for homicide"]

    User → *"Protection against arbitrary arrest under Indian Constitution"*  
    Keywords: ["arbitrary arrest", "Article 22", "personal liberty"]

    User → *"Police tortured a suspect to make him confess"*  
    Keywords: ["custodial torture", "forced confession", "Section 330", "police abuse"]

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
                logger.warning(f"⚠️ Only {len(keywords)} keyword(s) generated, should be 1-4")
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


@tool
def enhanced_cross_domain_legal_search(query: str) -> str:
    """
    🚀 PRIORITY TOOL for complex legal queries that span MULTIPLE legal domains or require cross-referencing.
    
    **USE THIS TOOL FIRST when queries involve ANY of these patterns:**
    - Questions about constitutional rights AND criminal law together
    - Queries mentioning BOTH "Constitution" AND "IPC" or "criminal"
    - Questions about "fundamental rights" AND their "restrictions" or "violations"
    - Queries asking how different areas of law "interact", "balance", or "conflict"
    - Questions about constitutional protections AND their legal consequences
    - Any query needing information from BOTH constitutional and criminal law
    
    **Example trigger phrases:**
    - "constitutional protections...and IPC provisions"
    - "fundamental rights...balance with criminal law"
    - "Article 19...interact with...hate speech and defamation"
    - "constitutional freedoms conflict with IPC sections"
    
    This tool automatically searches both Constitution and IPC databases and intelligently fuses 
    results with advanced cross-domain analysis. Much more effective than using separate tools.
    
    ⚠️ DO NOT use separate search_constitution + search_ipc tools if this tool applies.
    """
    logger.info(f"🔍 ENHANCED TOOL: enhanced_cross_domain_legal_search called with query: '{query[:100]}...'")
    
    # Get both database instances
    constitution_db = get_constitution_db()
    ipc_db = get_ipc_db()
    
    if not constitution_db and not ipc_db:
        error_msg = "Neither Constitution nor IPC database is available. Please check database configuration."
        logger.error(f"❌ {error_msg}")
        return error_msg
    
    try:
        logger.info("🚀 Starting enhanced cross-domain search across multiple databases...")
        
        all_enhanced_results = []
        search_summary = {}
        
        # Search Constitution database if available
        if constitution_db:
            logger.info("📜 Performing enhanced search on Constitution database...")
            try:
                const_results = constitution_db.enhanced_cross_domain_search(query, top_k=3)
                
                # Add database source to results
                for result in const_results:
                    result['source_database'] = 'Constitution'
                
                all_enhanced_results.extend(const_results)
                search_summary['constitution_results'] = len(const_results)
                logger.info(f"✅ Constitution enhanced search: {len(const_results)} results")
                
            except Exception as e:
                logger.error(f"❌ Error in Constitution enhanced search: {e}")
                search_summary['constitution_error'] = str(e)
        
        # Search IPC database if available
        if ipc_db:
            logger.info("⚖️ Performing enhanced search on IPC database...")
            try:
                ipc_results = ipc_db.enhanced_cross_domain_search(query, top_k=3)
                
                # Add database source to results
                for result in ipc_results:
                    result['source_database'] = 'IPC'
                
                all_enhanced_results.extend(ipc_results)
                search_summary['ipc_results'] = len(ipc_results)
                logger.info(f"✅ IPC enhanced search: {len(ipc_results)} results")
                
            except Exception as e:
                logger.error(f"❌ Error in IPC enhanced search: {e}")
                search_summary['ipc_error'] = str(e)
        
        # If no results from either database
        if not all_enhanced_results:
            logger.info("📭 No results found from enhanced cross-domain search")
            return "No relevant legal provisions found using enhanced cross-domain search. The query may be too specific or the databases may not contain relevant information."
        
        # Perform final cross-database result fusion
        logger.info("🔀 Performing final cross-database result fusion...")
        final_results = _perform_cross_database_fusion(all_enhanced_results, query, top_k=4)
        
        # Analyze cross-database coverage
        cross_db_analysis = _analyze_cross_database_coverage(final_results)
        
        logger.info(f"📊 Enhanced cross-domain search summary: {search_summary}")
        logger.info(f"🎯 Final results after cross-database fusion: {len(final_results)}")
        logger.info(f"🔗 Cross-database coverage: {cross_db_analysis}")
        
        # Format results for display
        formatted_results = []
        
        # Add search summary
        summary_lines = []
        summary_lines.append("=== ENHANCED CROSS-DOMAIN SEARCH SUMMARY ===")
        summary_lines.append(f"Query analyzed across {len([db for db in [constitution_db, ipc_db] if db])} legal databases")
        
        if search_summary.get('constitution_results'):
            summary_lines.append(f"Constitution database: {search_summary['constitution_results']} enhanced results")
        if search_summary.get('ipc_results'):
            summary_lines.append(f"IPC database: {search_summary['ipc_results']} enhanced results")
        
        summary_lines.append(f"Final fused results: {len(final_results)}")
        summary_lines.append(f"Cross-database relevance: {cross_db_analysis.get('cross_relevance_score', 0):.2f}")
        summary_lines.append("=" * 50)
        
        formatted_results.extend(summary_lines)
        formatted_results.append("")  # Empty line
        
        # Format individual results
        for i, result in enumerate(final_results):
            entity = result.get('entity', {})
            content = entity.get('text') or entity.get('content', 'No content available')
            
            # Enhanced metadata
            composite_score = result.get('composite_score', 0)
            domain_score = result.get('domain_score', 0)
            search_strategy = result.get('search_strategy', 'Unknown')
            source_db = result.get('source_database', 'Unknown')
            collection = result.get('collection', 'Unknown')
            distance = result.get('distance', 'N/A')
            
            formatted_results.append(
                f"Enhanced Result {i+1} (Composite Score: {composite_score:.3f}):\n"
                f"Source: {source_db} Database - {collection}\n"
                f"Search Strategy: {search_strategy}\n"
                f"Domain Relevance: {domain_score:.3f} | Original Distance: {distance:.4f}\n"
                f"Content: {content}\n"
            )
            
            logger.debug(f"📄 Enhanced Result {i+1}: {source_db} DB, Score: {composite_score:.3f}, Strategy: {search_strategy}")
        
        result_text = "\n".join(formatted_results)
        logger.info(f"✅ Enhanced cross-domain search completed, returning {len(final_results)} fused results")
        
        # Save enhanced search summary to log
        _save_enhanced_tool_log(query, final_results, search_summary, cross_db_analysis)
        
        return result_text
        
    except Exception as e:
        error_msg = f"Error in enhanced cross-domain legal search: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg


def _perform_cross_database_fusion(all_results: List[Dict[str, Any]], query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Perform intelligent fusion of results from multiple databases."""
    logger.info(f"🔀 Fusing results from {len(all_results)} cross-database results...")
    
    # Calculate cross-database relevance scores
    for result in all_results:
        entity = result.get('entity', {})
        content = entity.get('text') or entity.get('content', '')
        source_db = result.get('source_database', '')
        
        # Cross-database relevance bonus
        cross_db_bonus = _calculate_cross_database_bonus(content, query, source_db)
        
        # Update composite score with cross-database factor
        original_composite = result.get('composite_score', 0)
        enhanced_composite = original_composite * 0.8 + cross_db_bonus * 0.2
        
        result['enhanced_composite_score'] = enhanced_composite
        result['cross_db_bonus'] = cross_db_bonus
    
    # Remove cross-database duplicates
    unique_results = []
    seen_content_hashes = set()
    
    # Sort by enhanced composite score
    sorted_results = sorted(all_results, key=lambda x: x.get('enhanced_composite_score', 0), reverse=True)
    
    for result in sorted_results:
        entity = result.get('entity', {})
        content = entity.get('text') or entity.get('content', '')
        
        # Create content hash for duplicate detection across databases
        content_hash = hash(content[:150]) if content else hash(str(result.get('id')))
        
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            unique_results.append(result)
            
            if len(unique_results) >= top_k:
                break
    
    logger.info(f"🧹 Cross-database fusion: {len(all_results)} -> {len(unique_results)} unique results")
    return unique_results


def _calculate_cross_database_bonus(content: str, query: str, source_db: str) -> float:
    """Calculate bonus score for cross-database relevance."""
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Query spans multiple domains
    has_constitutional_terms = any(term in query_lower for term in ['constitutional', 'fundamental', 'rights', 'article'])
    has_criminal_terms = any(term in query_lower for term in ['criminal', 'punishment', 'ipc', 'section', 'offense'])
    
    # Content relevance to opposite domain
    if source_db == 'Constitution':
        opposite_domain_relevance = sum(1 for term in ['restriction', 'liable', 'punishment', 'violation'] if term in content_lower)
    else:  # IPC
        opposite_domain_relevance = sum(1 for term in ['constitutional', 'fundamental', 'rights', 'protection'] if term in content_lower)
    
    # Base bonus for cross-domain queries
    cross_domain_bonus = 0.1 if has_constitutional_terms and has_criminal_terms else 0.05
    
    # Relevance bonus
    relevance_bonus = min(opposite_domain_relevance * 0.05, 0.15)
    
    return cross_domain_bonus + relevance_bonus


def _analyze_cross_database_coverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze coverage across different databases and legal domains."""
    analysis = {
        'total_results': len(results),
        'database_distribution': {},
        'strategy_distribution': {},
        'cross_relevance_score': 0.0,
        'domain_balance': 'unknown'
    }
    
    if not results:
        return analysis
    
    # Count database sources
    const_count = sum(1 for r in results if r.get('source_database') == 'Constitution')
    ipc_count = sum(1 for r in results if r.get('source_database') == 'IPC')
    
    analysis['database_distribution'] = {
        'constitution': const_count,
        'ipc': ipc_count,
        'constitution_percentage': (const_count / len(results)) * 100,
        'ipc_percentage': (ipc_count / len(results)) * 100
    }
    
    # Count search strategies
    strategy_counts = {}
    for result in results:
        strategy = result.get('search_strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    analysis['strategy_distribution'] = strategy_counts
    
    # Calculate cross-relevance score
    if const_count > 0 and ipc_count > 0:
        balance_factor = min(const_count, ipc_count) / max(const_count, ipc_count)
        coverage_factor = min(const_count, ipc_count) / len(results)
        analysis['cross_relevance_score'] = balance_factor * coverage_factor
        analysis['domain_balance'] = 'balanced' if balance_factor > 0.5 else 'imbalanced'
    
    return analysis


def _save_enhanced_tool_log(query: str, results: List[Dict[str, Any]], search_summary: Dict[str, Any], 
                           cross_db_analysis: Dict[str, Any]):
    """Save enhanced tool execution results to log file."""
    try:
        import os
        from datetime import datetime
        
        # Create the correct directory path
        log_dir = os.path.join("agent_system", "admin", "scripts", "generated")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "enhanced_tool_results.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure the file can be created
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*120}\n")
            f.write(f"ENHANCED CROSS-DOMAIN TOOL EXECUTION LOG\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Query: '{query}'\n")
            f.write(f"{'='*120}\n\n")
            
            # Search summary
            f.write(f"CROSS-DATABASE SEARCH SUMMARY:\n")
            f.write(f"Constitution Results: {search_summary.get('constitution_results', 'N/A')}\n")
            f.write(f"IPC Results: {search_summary.get('ipc_results', 'N/A')}\n")
            f.write(f"Final Fused Results: {len(results)}\n\n")
            
            # Cross-database analysis
            f.write(f"CROSS-DATABASE ANALYSIS:\n")
            f.write(f"Database Distribution: {cross_db_analysis.get('database_distribution', {})}\n")
            f.write(f"Strategy Distribution: {cross_db_analysis.get('strategy_distribution', {})}\n")
            f.write(f"Cross-Relevance Score: {cross_db_analysis.get('cross_relevance_score', 0):.3f}\n")
            f.write(f"Domain Balance: {cross_db_analysis.get('domain_balance', 'unknown')}\n\n")
            
            # Detailed results
            f.write(f"ENHANCED RESULTS DETAILS:\n")
            for i, result in enumerate(results, 1):
                entity = result.get('entity', {})
                content = entity.get('text') or entity.get('content', 'No content available')
                
                f.write(f"Result {i}:\n")
                f.write(f"  Source Database: {result.get('source_database', 'Unknown')}\n")
                f.write(f"  Collection: {result.get('collection', 'Unknown')}\n")
                f.write(f"  Enhanced Composite Score: {result.get('enhanced_composite_score', 0):.4f}\n")
                f.write(f"  Original Composite Score: {result.get('composite_score', 0):.4f}\n")
                f.write(f"  Cross-DB Bonus: {result.get('cross_db_bonus', 0):.4f}\n")
                f.write(f"  Domain Score: {result.get('domain_score', 0):.4f}\n")
                f.write(f"  Search Strategy: {result.get('search_strategy', 'Unknown')}\n")
                f.write(f"  Query Variant: {result.get('query_variant', 'Unknown')[:60]}...\n")
                f.write(f"  Content: {content[:400]}...\n")
                f.write(f"{'-'*60}\n")
            
            f.write(f"\n{'='*120}\n\n")
        
        logger.info(f"📝 Enhanced tool log saved to: {log_file}")
            
    except Exception as e:
        logger.error(f"❌ Error saving enhanced tool log: {e}")
        # Try alternative path as fallback
        try:
            fallback_dir = "generated"
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_file = os.path.join(fallback_dir, "enhanced_tool_results.log")
            
            with open(fallback_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Enhanced tool executed for query: {query[:100]}...\n")
                f.write(f"Results: {len(results)} items\n")
                f.write(f"Summary: {search_summary}\n\n")
            
            logger.info(f"📝 Enhanced tool log saved to fallback location: {fallback_file}")
        except Exception as fallback_error:
            logger.error(f"❌ Failed to save to fallback location: {fallback_error}")


# List of all tools
tools = [generate_keywords, search_constitution, search_ipc, predict_punishment, enhanced_cross_domain_legal_search]
logger.info(f"🛠️ Tools module updated with {len(tools)} tools: {[tool.name for tool in tools]}")

