#!/usr/bin/env python3
"""
Test script to demonstrate enhanced insight collection and storage
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.langgraph_agent import LangGraphResearchAgent
from agent.knowledge_graph import get_knowledge_graph_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_insight_collection():
    """Test the enhanced insight collection capabilities"""
    
    # Initialize components
    agent = LangGraphResearchAgent()
    kg = get_knowledge_graph_manager()
    
    print("🔬 Testing Enhanced Insight Collection")
    print("=" * 50)
    
    # Test 1: Research request that should generate multiple insights
    print("\n📚 Test 1: Research Request")
    print("-" * 30)
    
    research_request = "Find papers about transformer attention mechanisms in neural networks"
    print(f"Request: {research_request}")
    
    try:
        result = await agent.process_request(research_request, "test_session_1", "research_test")
        print(f"✅ Research completed: {result.get('intent', 'unknown')}")
        print(f"Response preview: {result.get('response', '')[:200]}...")
    except Exception as e:
        print(f"❌ Research failed: {str(e)}")
    
    # Test 2: Check what insights were stored
    print("\n🧠 Test 2: Stored Insights Check")
    print("-" * 30)
    
    insights = kg.get_research_insights("transformer", limit=10)
    print(f"Found {len(insights)} insights about 'transformer':")
    
    for i, insight in enumerate(insights[:3], 1):
        print(f"\n  {i}. {insight.get('insight', '')[:150]}...")
        print(f"     Type: {insight.get('context', {}).get('insight_type', 'unknown')}")
        print(f"     Score: {insight.get('relevance_score', 0):.2f}")
    
    # Test 3: Knowledge query that should generate insights
    print("\n🔍 Test 3: Knowledge Query")
    print("-" * 30)
    
    knowledge_request = "What are the key findings about neural network attention mechanisms?"
    print(f"Query: {knowledge_request}")
    
    try:
        result = await agent.process_request(knowledge_request, "test_session_2", "knowledge_test")
        print(f"✅ Knowledge query completed: {result.get('intent', 'unknown')}")
        print(f"Response preview: {result.get('response', '')[:200]}...")
    except Exception as e:
        print(f"❌ Knowledge query failed: {str(e)}")
    
    # Test 4: Check insights again
    print("\n📊 Test 4: Updated Insights Check")
    print("-" * 30)
    
    insights = kg.get_research_insights("attention", limit=10)
    print(f"Found {len(insights)} insights about 'attention':")
    
    insight_types = {}
    for insight in insights:
        insight_type = insight.get('context', {}).get('insight_type', 'unknown')
        insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
    
    print("Insight breakdown by type:")
    for insight_type, count in insight_types.items():
        print(f"  - {insight_type}: {count}")
    
    # Test 5: Search all memories to see what's stored
    print("\n💾 Test 5: All Memory Types")
    print("-" * 30)
    
    all_memories = kg.get_all_memories(limit=20)
    memory_types = {}
    for memory in all_memories:
        memory_type = memory.get('metadata', {}).get('type', 'unknown')
        memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
    
    print(f"Total memories: {len(all_memories)}")
    print("Memory breakdown by type:")
    for memory_type, count in memory_types.items():
        print(f"  - {memory_type}: {count}")
    
    print("\n🎉 Insight collection testing complete!")
    print("The agent should now be generating and storing insights from:")
    print("  1. Individual paper analysis")
    print("  2. Research synthesis") 
    print("  3. Knowledge queries")

if __name__ == "__main__":
    asyncio.run(test_insight_collection())