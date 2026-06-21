from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()



model = ChatOpenAI()


parser = JsonOutputParser()

# 1st prompt -> detailed report
template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)