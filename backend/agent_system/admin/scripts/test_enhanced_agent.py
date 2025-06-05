"""
Test script to verify the enhanced cross-domain search functionality
and demonstrate the improved search capabilities for complex legal queries.
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_system.ai_agent.agent_graph import run_agent
from agent_system.utils.Constants import Constants

def test_enhanced_search_functionality():
    """Test the enhanced search functionality with complex legal queries."""
    
    print("=" * 80)
    print("TESTING ENHANCED CROSS-DOMAIN SEARCH FUNCTIONALITY")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test queries designed to trigger the enhanced search tool
    test_queries = [
        {
            "name": "Complex Constitutional-Criminal Query",
            "query": """What are the constitutional protections for freedom of speech and expression under Article 19, 
            and how do they interact with IPC provisions on hate speech and defamation? Specifically, 
            what are the reasonable restrictions on free speech, and what are the potential legal 
            consequences for violating these restrictions?""",
            "expected_tool": "enhanced_cross_domain_legal_search"
        },
        {
            "name": "Cross-Domain Rights and Restrictions",
            "query": """How do fundamental rights under the Constitution balance with criminal law provisions? 
            What happens when constitutional freedoms conflict with IPC sections on public order and morality?""",
            "expected_tool": "enhanced_cross_domain_legal_search"
        },
        # {
        #     "name": "Simple Constitutional Query",
        #     "query": "What is Article 21 of the Indian Constitution?",
        #     "expected_tool": "search_constitution"
        # },
        # {
        #     "name": "Simple Criminal Query", 
        #     "query": "What is the punishment for theft under IPC?",
        #     "expected_tool": "search_ipc"
        # }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"{'='*60}")
        print(f"Query: {test_case['query'][:100]}...")
        print(f"Expected to use: {test_case['expected_tool']}")
        print("-" * 60)
        
        try:
            print("🚀 Running agent...")
            response = run_agent(test_case['query'])
            
            print("✅ Agent response received")
            print(f"📏 Response length: {len(response)} characters")
            
            # Check if response indicates enhanced search was used
            enhanced_indicators = [
                "ENHANCED CROSS-DOMAIN SEARCH",
                "enhanced results",
                "Cross-database relevance",
                "domain relevance scoring",
                "Composite Score"
            ]
            
            used_enhanced = any(indicator in response for indicator in enhanced_indicators)
            
            results.append({
                "test_name": test_case['name'],
                "query": test_case['query'][:100] + "...",
                "expected_tool": test_case['expected_tool'],
                "used_enhanced": used_enhanced,
                "response_length": len(response),
                "success": True
            })
            
            print(f"🔍 Enhanced search detected: {'Yes' if used_enhanced else 'No'}")
            print(f"📊 Response preview: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Error in test {i}: {str(e)}")
            results.append({
                "test_name": test_case['name'],
                "query": test_case['query'][:100] + "...",
                "expected_tool": test_case['expected_tool'],
                "used_enhanced": False,
                "response_length": 0,
                "success": False,
                "error": str(e)
            })
        
        print("-" * 60)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = sum(1 for r in results if r['success'])
    enhanced_tests = sum(1 for r in results if r.get('used_enhanced', False))
    
    print(f"Total tests: {len(test_queries)}")
    print(f"Successful tests: {successful_tests}")
    print(f"Tests using enhanced search: {enhanced_tests}")
    print()
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        enhanced_status = "🔍 Enhanced" if result.get('used_enhanced', False) else "📝 Standard"
        
        print(f"{i}. {result['test_name']}: {status} {enhanced_status}")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   Response length: {result['response_length']} chars")
        print()
    
    print("📁 Check the following log files for detailed results:")
    print("   - generated/agent_execution.log")
    print("   - generated/agent_conversations.log")
    print("   - generated/enhanced_search_results.log")
    print("   - generated/enhanced_tool_results.log")
    print("   - generated/vector_db_results.log")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results

def analyze_log_files():
    """Analyze the generated log files to see search performance."""
    print("\n" + "=" * 80)
    print("LOG FILE ANALYSIS")
    print("=" * 80)
    
    log_files = [
        "generated/enhanced_search_results.log",
        "generated/enhanced_tool_results.log", 
        "generated/vector_db_results.log",
        "generated/agent_conversations.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.count('\n')
                size_kb = len(content) / 1024
                
                print(f"📄 {log_file}:")
                print(f"   Size: {size_kb:.1f} KB")
                print(f"   Lines: {lines}")
                
                # Count enhanced search entries
                if "enhanced" in log_file.lower():
                    enhanced_searches = content.count("ENHANCED CROSS-DOMAIN SEARCH")
                    print(f"   Enhanced searches logged: {enhanced_searches}")
                
            except Exception as e:
                print(f"❌ Error reading {log_file}: {e}")
        else:
            print(f"📄 {log_file}: Not found")
        print()

if __name__ == "__main__":
    print("Starting enhanced agent functionality test...")
    
    try:
        Constants.set_env_variables()
        print("Environment variables set")
        
        Constants.check_env_variables()
        print("Environment variables checked")
        
        # Run the tests
        test_results = test_enhanced_search_functionality()
        
        # Analyze log files
        analyze_log_files()
        
        print("\n🎉 Testing completed! Check the log files for detailed search analysis.")
        
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        import traceback
        traceback.print_exc() 