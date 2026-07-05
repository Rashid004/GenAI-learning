from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# load OPENAI_API_KEY from .env
load_dotenv()

# LangChain documents for IPL players
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

embeddings = OpenAIEmbeddings()

# from_documents already indexes docs, so build store here only (no separate add_documents needed)
vector_store = FAISS.from_documents(docs, embeddings)

# similarity search
results = vector_store.similarity_search("Who among these are a bowler?", k=2)
print("Similarity search results:")
for doc in results:
    print(doc.page_content, doc.metadata)

# similarity search with relevance score
results_with_scores = vector_store.similarity_search_with_score("Who among these are a bowler?", k=2)
print("\nSimilarity search with score:")
for doc, score in results_with_scores:
    print(score, doc.page_content, doc.metadata)

# metadata filtering (query must be non-empty for embedding to be meaningful)
filtered_results = vector_store.similarity_search_with_score(
    query="captain", filter={"team": "Mumbai Indians"}
)
print("\nFiltered results (Mumbai Indians):")
for doc, score in filtered_results:
    print(score, doc.page_content, doc.metadata)

# FAISS has no update_document method - update = delete old id + add new doc
doc2_id = vector_store.index_to_docstore_id[1]
updated_doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure. He is also a part-time bowler.",
    metadata={"team": "Mumbai Indians"}
)
vector_store.delete([doc2_id])
vector_store.add_documents([updated_doc2])
print("\nDocument updated.")
