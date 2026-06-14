from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen2.5-72B-Instruct",
  task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# result1 = model.invoke("What is the capital of India")
# result2 = model.invoke("What is the capital of France")
result3 = model.invoke("Who is the prime minister of india?")

# print(result1.content)
# print(result2.content)
print(result3.content)





