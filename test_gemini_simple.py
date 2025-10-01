"""
Simple Gemini API test
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("GEMINI API CONNECTION TEST")
print("=" * 60)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
print(f"✓ API Key loaded: {api_key[:10]}..." if api_key else "✗ API Key not found")

if not api_key:
    print("\n❌ Please set GEMINI_API_KEY in .env file")
    exit(1)

# Configure Gemini
print("\n[1/3] Configuring Gemini API...")
genai.configure(api_key=api_key)
print("✓ Configuration complete")

# Try to create a model
print("\n[2/3] Creating model instance...")
try:
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    print("✓ Model instance created: gemini-2.0-flash-exp")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    exit(1)

# Test generation
print("\n[3/3] Testing generation (10 second timeout)...")
try:
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Request took too long")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

    response = model.generate_content("Say hello in one word")

    signal.alarm(0)  # Cancel alarm

    if response and response.text:
        print(f"✓ Response received: '{response.text}'")
        print("\n" + "=" * 60)
        print("✅ SUCCESS - Gemini API is working!")
        print("=" * 60)
    else:
        print("✗ Empty response received")

except TimeoutError:
    print("✗ Request timed out (>10 seconds)")
    print("\n💡 Possible issues:")
    print("   - Slow internet connection")
    print("   - Firewall blocking Google AI services")
    print("   - VPN/proxy interference")
except Exception as e:
    print(f"✗ Generation failed: {str(e)[:200]}")
    print("\n💡 Possible issues:")
    print("   - Invalid API key")
    print("   - API quota exceeded")
    print("   - Network connectivity issues")