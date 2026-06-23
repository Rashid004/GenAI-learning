from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
  return len(text.split())

prompt = PromptTemplate(
  template="Generate a joke about {topic}",
  input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
  'joke': RunnablePassthrough(),
  'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)