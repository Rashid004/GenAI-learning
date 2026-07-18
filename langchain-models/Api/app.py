from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from dotenv import load_dotenv

import uvicorn
import os

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI(
  title="Langchain Server",
   version="1.0",
    decsription="A simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)
model=ChatOpenAI()

prompt = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")

add_routes(
    app,
    prompt | model,
    path="/essay"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)