#!/usr/bin/env python3
"""
Comprehensive test script to verify all components work properly
"""

def test_imports():
    """Test that all critical imports work"""
    print("Testing imports...")

    try:
        # Test basic ML imports
        import pandas as pd
        import numpy as np
        import requests
        print("✅ Basic libraries imported successfully")

        # Test transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        print("✅ Transformers and torch imported successfully")

        # Test ML libraries
        from sklearn.ensemble import RandomForestRegressor
        import ta
        print("✅ ML libraries imported successfully")

        # Test langchain
        from langchain_huggingface import HuggingFacePipeline
        from langchain.prompts import PromptTemplate
        print("✅ LangChain imported successfully")

        # Test embeddings
        from sentence_transformers import SentenceTransformer
        import faiss
        print("✅ Embedding libraries imported successfully")

        # Test API clients
        import finnhub
        print("✅ API clients imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_market_analysis_engine():
    """Test the market analysis engine"""
    print("\nTesting Market Analysis Engine...")

    try:
        import sys
        import os
        sys.path.append('/Volumes/D/RAGZ-C/RAG-Market_analysis_test')

        from ML_e import AdvancedMarketAnalysisEngine

        # Test initialization using config
        from config import Config
        Config.validate_api_keys()  # Ensure API key is available
        engine = AdvancedMarketAnalysisEngine(Config.FINNHUB_API_KEY)
        print("✅ Market analysis engine initialized")

        # Test basic functionality (without actual API call)
        assert hasattr(engine, 'comprehensive_analysis_with_prediction')
        assert hasattr(engine, 'format_advanced_analysis_for_llm')
        print("✅ Required methods exist")

        return True

    except Exception as e:
        print(f"❌ Market analysis engine error: {e}")
        return False

def test_langchain_files():
    """Test updated langchain files"""
    print("\nTesting LangChain files...")

    test_files = [
        '/Volumes/D/RAGZ-C/RAG-zul-general/langchain_llm2.py',
        '/Volumes/D/RAGZ-C/RAG-zuin_llm_huggingface_code_with_mods/langchain_llm2.py'
    ]

    for file_path in test_files:
        try:
            print(f"Checking syntax of {file_path}...")
            with open(file_path, 'r') as f:
                code = f.read()

            # Check for deprecated imports
            if 'langchain_community.llms' in code:
                print(f"⚠️ {file_path} still has deprecated langchain_community import")
            elif 'LLMChain' in code:
                print(f"⚠️ {file_path} still uses deprecated LLMChain")
            elif 'chain.run(' in code:
                print(f"⚠️ {file_path} still uses deprecated chain.run()")
            else:
                print(f"✅ {file_path} looks good")

        except Exception as e:
            print(f"❌ Error checking {file_path}: {e}")

def test_integration_syntax():
    """Test Integration.py syntax without loading heavy models"""
    print("\nTesting Integration.py syntax...")

    try:
        with open('/Volumes/D/RAGZ-C/RAG-Market_analysis_test/Integration.py', 'r') as f:
            code = f.read()

        # Check for proper import
        if 'from ML_e import AdvancedMarketAnalysisEngine' in code:
            print("✅ Correct import statement")
        else:
            print("❌ Import statement issue")

        # Check for proper token decoding
        if 'new_tokens = outputs[0][input_length:]' in code:
            print("✅ Proper token decoding implemented")
        else:
            print("❌ Token decoding issue")

        # Compile check
        compile(code, '/Volumes/D/RAGZ-C/RAG-Market_analysis_test/Integration.py', 'exec')
        print("✅ Integration.py compiles successfully")

        return True

    except SyntaxError as e:
        print(f"❌ Syntax error in Integration.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking Integration.py: {e}")
        return False

def test_simple_langchain():
    """Test a simple langchain example that doesn't require heavy models"""
    print("\nTesting simple LangChain functionality...")

    try:
        from langchain.prompts import PromptTemplate

        # Test prompt creation
        prompt = PromptTemplate(
            template="Hello {name}, how are you?",
            input_variables=["name"]
        )

        formatted = prompt.format(name="World")
        assert "Hello World" in formatted
        print("✅ Basic LangChain prompt functionality works")

        return True

    except Exception as e:
        print(f"❌ LangChain test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE COMPONENT TEST")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Market Analysis Engine", test_market_analysis_engine),
        ("LangChain Files", test_langchain_files),
        ("Integration Syntax", test_integration_syntax),
        ("Simple LangChain", test_simple_langchain),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! The system should work properly.")
    elif passed >= len(results) * 0.8:
        print("\n⚠️ Most tests passed. Minor issues may exist but system should mostly work.")
    else:
        print("\n❌ Multiple issues detected. System may not work properly.")

if __name__ == "__main__":
    main()