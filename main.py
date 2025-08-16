import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]




llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a research assistant that will help generate a research paper.
        Answer the user's query and use necessary tools to answer the question.
        Wrap the output in this format and provide no other text:
        {format_instructions}
        """
    ),
    (
        "placeholder",
        "{chat_history}"
    ),
    (
        "human",
        "{query}"
    ),
    (
        "placeholder",
        "{agent_scratchpad}"
    )
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

# The executor runs the agent, allowing it to:
# 1. Think about the query
# 2. Decide which tools to use (search_tool, save_tool)
# 3. Execute tools in sequence
# 4. Format the final response
# 5. Save the response to a text file
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your query: ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response: ", e, "\n", raw_response.get("output"))




