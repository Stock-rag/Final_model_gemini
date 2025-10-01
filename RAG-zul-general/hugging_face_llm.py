from transformers import pipeline

# Load a free Hugging Face model (FLAN-T5 is small & fast)
generator = pipeline("text2text-generation", model="google/flan-t5-base")

response = generator("Explain large language model in one sentence")
print(response[0]["generated_text"])
