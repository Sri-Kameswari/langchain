from dotenv import load_dotenv
from langchain.chains.question_answering.map_rerank_prompt import output_parser

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=['input', 'agent_scratchpad', 'tool_names']
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(tools=tools, agent=agent, verbose=True)
extract_output = RunnableLambda(lambda x: x['output'])
parse_output = RunnableLambda(lambda x: output_parser.parse(text=x))
chain = agent_executor | extract_output | parse_output

def main():
    print("output_parser exists:", 'output_parser' in globals())
    result = chain.invoke(
        input={
            "input": "search for 5 job openings for java developer in hyderabad on linkedin and list their details"
        }
    )
    print(result)

if __name__ == "__main__":
    main()