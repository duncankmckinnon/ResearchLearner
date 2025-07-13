#!/usr/bin/env python3
"""
Test script to demonstrate tool-based execution flow for knowledge graph operations
"""

import asyncio
import logging
from agent.knowledge_tools import get_knowledge_tools, get_knowledge_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tool_execution():
    """Test the tool-based execution flow"""
    
    print("🔧 Testing Tool-Based Knowledge Graph Execution")
    print("=" * 60)
    
    # Test 1: List all available tools
    print("\n📋 Test 1: Available Tools")
    print("-" * 30)
    
    tools = get_knowledge_tools()
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Test 2: Test individual tool execution
    print("\n🔍 Test 2: Search Knowledge Tool")
    print("-" * 30)
    
    search_tool = get_knowledge_tool("search_knowledge")
    if search_tool:
        try:
            results = await search_tool._arun("neural networks", limit=3, user_id="test_tools")
            print(f"✅ Search completed: found {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. {result.get('content', '')[:100]}...")
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
    
    # Test 3: Test adding paper
    print("\n📄 Test 3: Add Research Paper Tool")
    print("-" * 30)
    
    add_paper_tool = get_knowledge_tool("add_research_paper")
    if add_paper_tool:
        try:
            test_paper = {
                "title": "Test Paper: Tool-Based Knowledge Management",
                "authors": ["Tool Tester", "Knowledge Graph"],
                "abstract": "A test paper to demonstrate tool-based knowledge graph operations.",
                "paper_id": "test-2024-001",
                "categories": ["cs.AI", "cs.LG"],
                "content": "This is a test paper for demonstrating async tool execution."
            }
            
            success = await add_paper_tool._arun(test_paper, user_id="test_tools")
            print(f"✅ Paper added: {success}")
        except Exception as e:
            print(f"❌ Paper addition failed: {str(e)}")
    
    # Test 4: Test adding insight
    print("\n💡 Test 4: Add Research Insight Tool")
    print("-" * 30)
    
    add_insight_tool = get_knowledge_tool("add_research_insight")
    if add_insight_tool:
        try:
            test_insight = "Tool-based execution provides better visibility into agent operations and allows for async knowledge graph access."
            test_context = {
                "insight_type": "tool_testing",
                "execution_mode": "async",
                "test_session": "tool_validation"
            }
            
            success = await add_insight_tool._arun(
                test_insight, 
                "tool execution", 
                test_context, 
                user_id="test_tools"
            )
            print(f"✅ Insight added: {success}")
        except Exception as e:
            print(f"❌ Insight addition failed: {str(e)}")
    
    # Test 5: Test getting insights
    print("\n🧠 Test 5: Get Research Insights Tool")
    print("-" * 30)
    
    get_insights_tool = get_knowledge_tool("get_research_insights")
    if get_insights_tool:
        try:
            insights = await get_insights_tool._arun("tool execution", limit=5, user_id="test_tools")
            print(f"✅ Retrieved {len(insights)} insights")
            for i, insight in enumerate(insights[:2], 1):
                print(f"  {i}. {insight.get('insight', '')[:100]}...")
                print(f"     Type: {insight.get('context', {}).get('insight_type', 'unknown')}")
        except Exception as e:
            print(f"❌ Insight retrieval failed: {str(e)}")
    
    # Test 6: Test knowledge summary
    print("\n📊 Test 6: Knowledge Summary Tool")
    print("-" * 30)
    
    summary_tool = get_knowledge_tool("get_knowledge_summary")
    if summary_tool:
        try:
            summary = await summary_tool._arun("tool execution", user_id="test_tools")
            print(f"✅ Summary retrieved:")
            print(f"  - Papers: {summary.get('total_papers', 0)}")
            print(f"  - Insights: {summary.get('total_insights', 0)}")
            print(f"  - Knowledge items: {summary.get('total_knowledge_items', 0)}")
        except Exception as e:
            print(f"❌ Summary retrieval failed: {str(e)}")
    
    # Test 7: Test related papers
    print("\n📚 Test 7: Related Papers Tool")
    print("-" * 30)
    
    papers_tool = get_knowledge_tool("get_related_papers")
    if papers_tool:
        try:
            papers = await papers_tool._arun("tool execution", limit=3, user_id="test_tools")
            print(f"✅ Retrieved {len(papers)} related papers")
            for i, paper in enumerate(papers[:2], 1):
                print(f"  {i}. {paper.get('title', 'Unknown title')}")
                print(f"     Source: {paper.get('source', 'unknown')}")
                print(f"     Score: {paper.get('relevance_score', 0):.2f}")
        except Exception as e:
            print(f"❌ Papers retrieval failed: {str(e)}")
    
    print("\n🎉 Tool-based execution testing complete!")
    print("\nKey Benefits Demonstrated:")
    print("  ✓ Async execution of knowledge graph operations")
    print("  ✓ Proper tool encapsulation with input/output schemas")
    print("  ✓ Clear execution visibility through logging")
    print("  ✓ Non-blocking operations for better performance")
    print("  ✓ Standardized tool interface for LangGraph integration")

if __name__ == "__main__":
    asyncio.run(test_tool_execution())