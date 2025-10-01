from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
# Load the tokenizer and model
model_name = "distributed/llama-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Prepare your input text
input_text = "Hello, how can I assist you today?"

# Tokenize and generate a response
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
