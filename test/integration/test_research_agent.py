#!/usr/bin/env python3
"""
Test script for the research agent with ArXiv MCP server integration
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.langgraph_agent import LangGraphResearchAgent
from opentelemetry import trace
from agent.constants import PROJECT_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_research_agent")

# Initialize tracer for testing
tracer = trace.get_tracer(PROJECT_NAME)

async def test_research_agent():
    """Test the research agent functionality"""
    
    # Initialize the agent with trace provider for consistency
    agent = LangGraphResearchAgent(trace_provider=tracer)
    
    # Test cases
    test_cases = [
        {
            "name": "Research Request",
            "request": "Find papers about transformer architectures in machine learning",
            "expected_intent": "research"
        },
        {
            "name": "Knowledge Query",
            "request": "What do I know about neural networks?",
            "expected_intent": "knowledge_query"
        },
        {
            "name": "General Question",
            "request": "Hello, how are you?",
            "expected_intent": "general"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Request: {test_case['request']}")
        print(f"Expected Intent: {test_case['expected_intent']}")
        print(f"{'='*60}")
        
        try:
            # Process the request
            session_id = f"test_session_{i}"
            result = await agent.process_request(test_case['request'], session_id)
            
            # Print results
            print(f"✅ Success!")
            print(f"Detected Intent: {result.get('intent', 'Unknown')}")
            print(f"Plan: {result.get('plan', [])}")
            print(f"Response Preview: {result.get('response', 'No response')[:200]}...")
            
            # Check if research data was generated for research requests
            if test_case['expected_intent'] == 'research':
                research_data = result.get('research_data')
                if research_data:
                    search_results = research_data.get('search_results', {})
                    papers_found = search_results.get('papers_found', 0)
                    print(f"Papers Found: {papers_found}")
                    
                    analyzed_papers = research_data.get('analyzed_papers', [])
                    print(f"Papers Analyzed: {len(analyzed_papers)}")
                else:
                    print("⚠️  No research data generated")
            
            # Check if knowledge data was generated for knowledge queries
            if test_case['expected_intent'] == 'knowledge_query':
                knowledge_data = result.get('knowledge_data')
                if knowledge_data:
                    print(f"Knowledge Items Found: {len(knowledge_data.get('search_results', []))}")
                else:
                    print("⚠️  No knowledge data generated")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Test {i} failed: {str(e)}")

async def test_mcp_client():
    """Test the MCP client directly"""
    print(f"\n{'='*60}")
    print("Testing MCP Client Directly")
    print(f"{'='*60}")
    
    try:
        from agent.mcp_client import ResearchAgent
        
        research_agent = ResearchAgent()
        
        # Test research topic
        print("Testing research topic...")
        result = await research_agent.research_topic("machine learning", max_papers=3)
        
        print(f"Topic: {result.get('topic')}")
        print(f"Papers Found: {result.get('papers_found', 0)}")
        print(f"Papers Downloaded: {result.get('papers_downloaded', 0)}")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ MCP Client working!")
            
    except Exception as e:
        print(f"❌ MCP Client Error: {str(e)}")
        logger.error(f"MCP client test failed: {str(e)}")

async def test_knowledge_graph():
    """Test the knowledge graph functionality"""
    print(f"\n{'='*60}")
    print("Testing Knowledge Graph")
    print(f"{'='*60}")
    
    try:
        # Use the shared knowledge graph instance from the agent
        agent = LangGraphResearchAgent(trace_provider=tracer)
        kg = agent.knowledge_graph
        
        # Test adding an insight
        success = kg.add_research_insight(
            "Test insight about machine learning",
            "machine learning",
            {"test": True}
        )
        
        if success:
            print("✅ Added test insight to knowledge graph")
        else:
            print("⚠️  Failed to add insight (mem0 might not be properly configured)")
        
        # Test searching
        results = kg.search_knowledge("machine learning", limit=5)
        print(f"Knowledge search results: {len(results)} items found")
        
        if results:
            print("✅ Knowledge graph search working!")
        else:
            print("⚠️  No results from knowledge graph search")
            
    except Exception as e:
        print(f"❌ Knowledge Graph Error: {str(e)}")
        logger.error(f"Knowledge graph test failed: {str(e)}")

async def main():
    """Main test function"""
    print("🚀 Starting Research Agent Tests")
    print("=" * 80)
    
    # Test individual components first
    await test_mcp_client()
    await test_knowledge_graph() 
    
    # Test the full research agent
    await test_research_agent()
    
    print("\n🏁 Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())