from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

model = ChatOpenAI()

class Facts(BaseModel):
    name: str = Field(description="Topic name")
    facts: list[str] = Field(description="List of facts")
    summary: str = Field(description="Short summary")

# schema = [
#     ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
#     ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
#     ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
# ]

# parser = StructuredOutputParser.from_response_schemas(schema)


structured_model = model.with_structured_output(Facts)

# 1st prompt -> detailed report
# template = PromptTemplate(
#     template='Give me 3 fact about {topic} \n {format_instruction}',
#     input_variables=['topic'],
#     partial_variables={'format_instruction': parser.get_format_instructions()}
# )

# chain = template | model | parser

# result = chain.invoke({'topic':'black hole'})

# print(result)


result = structured_model.invoke(
    "Give me 3 facts about black holes"
)

# Use when you want Data in JSON format
print(result.model_dump())
