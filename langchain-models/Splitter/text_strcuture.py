from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial intelligence (AI) is technology that enables computers and machines to simulate human learning, comprehension, problem solving, decision making, creativity and autonomy.

Applications and devices equipped with AI can see and identify objects. They can understand and respond to human language. They can learn from new information and experience. They can make detailed recommendations to users and experts. They can act independently, replacing the need for human intelligence or intervention (a classic example being a self-driving car).
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=[""])

chunks = splitter.split_text(text)

print(chunks)

print(len(chunks))