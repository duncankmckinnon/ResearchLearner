#!/usr/bin/env python3
"""
Test script to debug mem0 configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_mem0_basic():
    """Test basic mem0 functionality"""
    try:
        from mem0 import Memory
        
        print("✅ mem0 import successful")
        
        # Test basic configuration
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "test_collection",
                    "path": "test_db"
                }
            }
        }
        
        print("🔧 Testing basic configuration...")
        memory = Memory.from_config(config)
        print("✅ Basic mem0 configuration successful")
        
        # Test adding a simple memory
        print("📝 Testing memory add...")
        result = memory.add("This is a test memory about machine learning", user_id="test_user")
        print(f"✅ Memory added: {result}")
        
        # Test searching
        print("🔍 Testing memory search...")
        results = memory.search("machine learning", user_id="test_user")
        print(f"✅ Search results: {len(results)} items found")
        for result in results:
            print(f"   - {result.get('memory', 'No content')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_mem0_with_openai():
    """Test mem0 with OpenAI configuration"""
    try:
        from mem0 import Memory
        
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  No OPENAI_API_KEY found, skipping OpenAI test")
            return True
            
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "test_collection_openai",
                    "path": "test_db_openai"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "temperature": 0.1
                }
            }
        }
        
        print("🔧 Testing OpenAI configuration...")
        memory = Memory.from_config(config)
        print("✅ OpenAI mem0 configuration successful")
        
        # Test adding a memory
        result = memory.add("Transformers are a type of neural network architecture", user_id="test_user")
        print(f"✅ Memory with OpenAI added: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI config error: {str(e)}")
        return False

def cleanup():
    """Clean up test files"""
    import shutil
    try:
        if os.path.exists("test_db"):
            shutil.rmtree("test_db")
        if os.path.exists("test_db_openai"):
            shutil.rmtree("test_db_openai")
        print("🧹 Cleaned up test files")
    except Exception as e:
        print(f"⚠️  Cleanup error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing mem0 Configuration")
    print("=" * 50)
    
    # Test basic functionality
    basic_success = test_mem0_basic()
    
    print("\n" + "=" * 50)
    
    # Test with OpenAI
    openai_success = test_mem0_with_openai()
    
    print("\n" + "=" * 50)
    print("🧹 Cleaning up...")
    cleanup()
    
    print("\n📊 Results:")
    print(f"   Basic mem0: {'✅ Working' if basic_success else '❌ Failed'}")
    print(f"   OpenAI integration: {'✅ Working' if openai_success else '❌ Failed'}")
    
    if basic_success and openai_success:
        print("\n🎉 mem0 is working correctly!")
    else:
        print("\n⚠️  mem0 needs configuration fixes")