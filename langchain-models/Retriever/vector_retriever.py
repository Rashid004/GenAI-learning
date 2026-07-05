from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Create embeddings for the documents
embeddings = OpenAIEmbeddings()

# Step 3: Create a Chroma vector store from the documents and embeddings
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name="my_collection")

# Step 4: Convert the vector store into a retriever with a specified number of results to return
retriever =  vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
