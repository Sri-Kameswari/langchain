from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

tools = [TavilySearch()]
llm = ChatOpenAI(model='gpt-4')
react_prompt = hub.pull('hwchase17/react')
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(tools=tools, agent=agent, verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input":"search for 5 job openings for java developer in hyderabad on linkedin and list their details"
        }
    )
    print(result)

if __name__ == "__main__":
    main()