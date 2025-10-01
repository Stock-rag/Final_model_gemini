"""
Test script for the updated Integration.py with Llama
"""
import sys
import os
sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')

async def test_integration():
    """Test the updated integration with Llama"""
    try:
        from Integration import StockRAGChatbot

        # Test API key from config
        from config import Config
        Config.validate_api_keys()
        API_KEY = Config.FINNHUB_API_KEY

        print("Initializing StockRAGChatbot with Llama...")
        chatbot = StockRAGChatbot(API_KEY)

        print("✅ Chatbot initialized successfully!")

        # Test basic functionality
        print("\nTesting basic chat functionality...")
        test_queries = [
            "Hello, how are you?",
            "What is AAPL stock?",
            "aapl stock",
        ]

        for query in test_queries:
            print(f"\nUser: {query}")
            try:
                response = await chatbot.chat(query)
                print(f"Assistant: {response}")  # Full response
                print(f"Response length: {len(response)} characters")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")

        print("\n✅ Integration test completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integration())