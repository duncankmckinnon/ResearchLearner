#!/usr/bin/env python3
"""
Test script for complete LangGraph agent workflow
"""

import asyncio
import logging
from ....agent.langgraph_agent import LangGraphResearchAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_workflow():
    """Test the complete LangGraph agent workflow"""
    print("🚀 Testing Complete LangGraph Agent Workflow")
    print("=" * 60)
    
    # Initialize agent
    agent = LangGraphResearchAgent()
    
    test_cases = [
        {
            "request": "What do I know about neural networks?",
            "context": "User wants to check existing knowledge",
            "expected_intent": "knowledge_query"
        },
        {
            "request": "Find papers about transformer architectures",
            "context": "User wants to research transformers", 
            "expected_intent": "research"
        },
        {
            "request": "Hello, how can you help me?",
            "context": "General greeting",
            "expected_intent": "general"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['expected_intent'].upper()}")
        print("-" * 40)
        print(f"Request: {test_case['request']}")
        print(f"Context: {test_case['context']}")
        
        try:
            # Process request through complete workflow
            result = await agent.process_request(
                user_request=test_case['request'],
                session_id=f"test_session_{i}",
                context=test_case['context']
            )
            
            print(f"✅ Intent detected: {result.get('intent', 'unknown')}")
            print(f"✅ Response generated: {len(result.get('response', ''))} characters")
            print(f"✅ Messages in conversation: {len(result.get('messages', []))}")
            
            # Show final response (truncated)
            response = result.get('response', 'No response')
            if len(response) > 200:
                response = response[:200] + "..."
            print(f"📝 Final Response: {response}")
            
        except Exception as e:
            print(f"❌ Error in test case {i}: {str(e)}")
            logger.error(f"Test case {i} failed: {str(e)}", exc_info=True)
    
    print("\n🎉 Complete workflow testing finished!")

if __name__ == "__main__":
    asyncio.run(test_complete_workflow())