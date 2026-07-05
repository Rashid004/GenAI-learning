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
# 5. Format retrieved docs into plain text for the prompt
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------
# 6. Prompt Template
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer only using the context.

Context:
{context}

Question:
{question}
""")


# ----------------------------
# 7. Output Parser
# ----------------------------
parser = StrOutputParser()


# ----------------------------
# 8. Runnable Chain (retriever -> prompt -> model -> parser)
# ----------------------------
chain = (
    RunnableParallel(
        context=retrieve | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | model
    | parser
)


# ----------------------------
# 9. Invoke — ask questions from the terminal until user quits
# ----------------------------
if __name__ == "__main__":
    print("Ask questions about the PDF. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}")
