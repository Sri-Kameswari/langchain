from typing import List, Union

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import render_text_description, BaseTool
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters in a string."""
    text = text.strip()
    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"No tool with name '{tool_name}' found")


def main():
    print("Hello from react-langchain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question (once all steps are completed)
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=",".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(temperature=0, stop_sequences=["Observation:", "Final Answer:"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt | llm | ReActSingleInputOutputParser()
    )

    agent_step = ""

    while not isinstance(agent, AgentFinish):

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length 'GIRAFFE' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction): #checking whether agent_step is an instance of type AgentAction
            tool_name = agent_step.tool
            print("tool name is " + tool_name)
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.invoke(tool_input)
            print(observation)

            intermediate_steps.append((agent_step, str(observation)))


        print(agent_step)

        #agent_step is not automatically moving to AgentFinish state. whyyyyy????

        if isinstance(agent_step, AgentFinish):
            print(agent_step.return_values)


if __name__ == "__main__":
    main()
