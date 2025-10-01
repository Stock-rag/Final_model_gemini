# langchain_llm_open.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# ---------------------------
# 1️⃣ Load an open LLaMA 2 model
# ---------------------------
model_name = "NousResearch/Llama-2-7b-hf"  # fully open variant

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token if not exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Changed from "cuda" to "auto" for better device compatibility
    torch_dtype=torch.float16
)

print("Model loaded successfully")


# #tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     device_map="auto",       # automatically uses GPU if available
# #     torch_dtype="auto",
# #     load_in_8bit=True      # automatically chooses FP16 or BF16 if supported
# #)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     device_map="auto",                      # Automatically assigns submodules to available devices (GPU + CPU)
# #     load_in_8bit=True,                      # Quantize model to 8-bit
# #     llm_int8_enable_fp32_cpu_offload=True, # Allow offloading some modules (in full precision) to CPU
# #     torch_dtype="auto"                      # Use FP16 or BF16 where supported
# # )
# from transformers import BitsAndBytesConfig
#
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,                    # or use load_in_4bit=True for lower RAM
#     llm_int8_enable_fp32_cpu_offload=True # enable CPU offload if GPU VRAM is low
# )
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     quantization_config=quantization_config,
#     torch_dtype="auto"
# )

# ---------------------------
# 2️⃣ Create a HuggingFace pipeline
# ---------------------------
from langchain_huggingface import HuggingFacePipeline #updated as old version is to be decrepated
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# 3️⃣ Define a LangChain prompt
# ---------------------------
prompt = PromptTemplate(
    input_variables=["concept"],
    template="Explain the concept of {concept} in one simple sentence."
)

# ---------------------------
# 4️⃣ Create a chain
# ---------------------------
chain = prompt | llm

# ---------------------------
# 5️⃣ Run the chain
# ---------------------------
concept = "Large Language Model"
result = chain.invoke({"concept": concept})
print("Result:\n", result)
