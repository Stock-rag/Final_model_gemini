"""
Check available Gemini models with your API key
"""
import google.generativeai as genai
import os

# Load API key
api_key = "AIzaSyCggEE5oZJAhDlWEzCST8AFj_aOjJDHN_g"
genai.configure(api_key=api_key)

print("🔍 Checking available Gemini models...")

try:
    # List all available models
    models = genai.list_models()

    print(f"\n✅ Found {len(list(models))} models:")
    models = genai.list_models()  # Call again since generator was consumed

    for model in models:
        print(f"📋 Model: {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Supported Methods: {model.supported_generation_methods}")
        print("-" * 50)

except Exception as e:
    print(f"❌ Error listing models: {e}")

# Try to find working models for generateContent
print("\n🧪 Testing models for generateContent...")

test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    "gemini-1.0-pro",
    "models/gemini-1.0-pro"
]

working_models = []

for model_name in test_models:
    try:
        print(f"Testing: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello")
        if response and response.text:
            print(f"✅ {model_name} - WORKS!")
            working_models.append(model_name)
        else:
            print(f"❌ {model_name} - No response")
    except Exception as e:
        print(f"❌ {model_name} - Error: {e}")

print(f"\n🎉 Working models: {working_models}")