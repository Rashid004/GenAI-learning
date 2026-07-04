"""
PDF Chatbot — RAG pipeline over a PDF using LangChain + FAISS + OpenAI.

Pipeline: load PDF -> split into chunks -> embed -> store in FAISS ->
retrieve relevant chunks -> answer question using only that context.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()


# ----------------------------
# 1. Model
# ----------------------------
model = ChatOpenAI(
    model="gpt-4o-mini"
)


# ----------------------------
# 2. Document Loader
# ----------------------------
loader = PyPDFLoader("data/python_book_sample.pdf")
documents = loader.load()


# ----------------------------
# 3. Text Splitter
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)


# ----------------------------
# 4. Embeddings + Vector Store
# ----------------------------
embeddings = OpenAIEmbeddings()
vectorStores = FAISS.from_documents(chunks, embeddings)
retrieve = vectorStores.as_retriever()


# ----------------------------
# 5. Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer only using the context.

Context:
{context}

Question:
{question}
""")


# ----------------------------
# 6. Output Parser
# ----------------------------
parser = StrOutputParser()


# ----------------------------
# 7. Runnable Chain (retriever -> prompt -> model -> parser)
# ----------------------------
chain = (
    RunnableParallel(
        context=retrieve,
        question=RunnablePassthrough(),
    )
    | prompt
    | model
    | parser
)


# ----------------------------
# 8. Invoke — ask a question from the terminal
# ----------------------------
if __name__ == "__main__":
    question = input("Ask a question about the PDF: ")
    answer = chain.invoke(question)
    print(answer)
