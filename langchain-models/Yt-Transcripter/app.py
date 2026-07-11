"""
YouTube Transcript Q&A — RAG pipeline over a video's transcript.

Pipeline: fetch transcript -> split into chunks -> embed + store in FAISS
-> retrieve relevant chunks for a question -> answer via LLM (RAG).
"""

import re

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
"""

# Matches the video ID out of youtu.be/, watch?v=, /embed/, or /shorts/ URLs,
# or accepts a bare 11-character ID.
_VIDEO_ID_PATTERN = re.compile(
    r"(?:youtu\.be/|v=|/embed/|/shorts/)([A-Za-z0-9_-]{11})|^([A-Za-z0-9_-]{11})$"
)


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from a YouTube URL or a bare ID."""
    match = _VIDEO_ID_PATTERN.search(url_or_id.strip())
    if not match:
        raise ValueError(f"Could not find a video ID in '{url_or_id}'.")
    return match.group(1) or match.group(2)


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    """Fetch a YouTube video's transcript and flatten it to plain text."""
    languages = languages or ["en"]
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except TranscriptsDisabled:
        raise RuntimeError(f"No captions available for video '{video_id}'.")
    return " ".join(snippet.text for snippet in fetched)


def build_vector_store(transcript: str) -> FAISS:
    """Split the transcript into chunks and index them in a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def format_docs(retrieved_docs) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_rag_chain(retriever):
    """Wire up the retrieval-augmented generation chain: retrieve -> prompt -> LLM -> parse."""
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    return parallel_chain | prompt | llm | parser


def main():
    """CLI entry point (kept for scripted/non-UI usage)."""
    video_id = "MIlDK1qQLaI"
    transcript = fetch_transcript(video_id)

    vector_store = build_vector_store(transcript)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = build_rag_chain(retriever)

    question = "Can you summarize the video?"
    answer = chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
