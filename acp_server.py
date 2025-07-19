import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from smolagents import LiteLLMModel
import os
load_dotenv()

server = Server()
client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],  ## Ensure correct absolute path
                "transport": "stdio",

            },

            # "weather": {
            #     "url": "http://localhost:8000/mcp",  # Ensure server is running here
            #     "transport": "streamable_http",
            # }

        }
    )
llm_model = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_API_KEY"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_ID"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0,
    )

@server.agent()
async def calculation_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    '''calculation_agent is a basic math calculation agent which perform addition and multiplication operation'''
    tools = await client.get_tools()
    tools = [tool for tool in tools if tool.name in ['add', 'multiply']]
    # print("tools:", tools)
    agent = initialize_agent(
        tools=tools,
        llm=llm_model,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    # math_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is 2+2?"}]})
    # math_response = await agent.arun('''{"messages": [{"role": "user", "content": "what is (3+5)*(12*3)?"}]}''')
    prompt = input[0].parts[0].content
    response = await agent.ainvoke(prompt)


    yield Message(parts=[MessagePart(content=str(response))])

# @server.agent()
# async def conversation_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
#     '''conversation_agent is responsible to perform general conversation with user'''
#     prompt = input[0].parts[0].content
#     response = await llm_model.ainvoke(prompt)
#
#     yield Message(parts=[MessagePart(content=str(response.content))])


@server.agent()
async def weather_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    '''This agent will be used to find weather of given state'''
    response='Not able to fetch weather report'
    tools = await client.get_tools()
    tools = [tool for tool in tools if tool.name in ['weather']]

    # prompt = input[0].parts[0].content


    if tools:
        # response = await tools[0].arun({'state':json.loads(prompt['input'])})
        response = await tools[0].arun({'state':'Delhi'})

    yield Message(parts=[MessagePart(content=str(response))])


@server.agent()
async def termination_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    '''termination_agent will be used to the end the conversation to terminate the converastion'''
    yield Message(parts=[MessagePart(content=str(input[0].parts[0].content))])


if __name__ == "__main__":
    server.run(port=8001)