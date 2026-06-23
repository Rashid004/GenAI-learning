from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
  template="Generate a joke about {topic}",
  input_variables=['topic']
)

model = ChatOpenAI()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
  input_variables=['text']
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
  'joke': RunnablePassthrough(),
  'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

print(final_chain.invoke({'topic':'Cricket'}))