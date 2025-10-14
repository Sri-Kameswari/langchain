from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()


def main():
    print("Hello from text-summariser!")
    information = """
    Harry, on his eleventh birthday, learns he is a wizard. He attends Hogwarts, a school of magic, where he receives guidance from the headmaster Albus Dumbledore and becomes friends with Ron Weasley and Hermione Granger. Harry learns that during his infancy, the Dark wizard Lord Voldemort murdered his parents but was unable to kill him as well.
    """

    summary_template = """
    Given the info {information} about the person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # temperature = 0 → deterministic, temperature = 1 → balanced creativity, temperature > 1 → more random/creative

    chain = summary_prompt_template | llm
    # output of 'summary_prompt_template' is passed as the input for 'llm'
    # the resulting chain is a runnable object

    response = chain.invoke(input={"information": information})
    #invoke takes list of messages and responds with single message
    print(response.content)

if __name__ == "__main__":
    main()
