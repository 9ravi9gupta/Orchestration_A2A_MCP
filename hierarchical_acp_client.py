import asyncio
from acp_sdk.client import Client
from smolagents import LiteLLMModel
from fastacp import AgentCollection, ACPCallingAgent
from colorama import Fore
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id=os.getenv("AZURE_MODEL_ID"),
                    api_base=os.getenv("AZURE_API_BASE"),
                    api_key=os.getenv("AZURE_API_KEY"),
                    api_version=os.getenv("AZURE_API_VERSION"),
                    temperature=float(os.getenv("AZURE_TEMPERATURE", "0.2"))
                    )


prompt_template2 = {
    "system_prompt":'''You are a supervisory agent that delegates tasks to specialized ACP agents.

############ Available agents: ############
{agents}

############ TASK INSTRUCTIONS ############
1. Analyze the user’s request and check if it has already been answered.
2. If needed, call one or more agents sequentially to gather required info.
3. Do NOT call any agent again if its result is already known from memory or user input.
4. After all needed info is available, summarize the answer clearly.
5. Then, call `termination_agent` with the full summary to end the conversation.

############ IMPORTANT NOTES ############
- You MUST trust the memory state and user-supplied information (e.g. “Observation: …”).
- Do NOT re-call an agent if its task has been fulfilled.
- If there’s any issue or error, call `termination_agent` with available info to stop execution.
'''}

async def run_calculation_workflow() -> None:
    async with Client(base_url="http://localhost:8001") as run_cal:
        # agents discovery
        agent_collection = await AgentCollection.from_acp(run_cal)
        acp_agents = {agent.name: {'agent':agent, 'client':client} for client, agent in agent_collection.agents}
        # print(acp_agents)
        # passing the agents as tools to ACPCallingAgent
        acpagent = ACPCallingAgent(acp_agents=acp_agents, model=model, prompt_templates=prompt_template2)
        result = await acpagent.run("give me weather of delhi?")
        print(Fore.YELLOW + f"Final result: {result}" + Fore.RESET)

if __name__ == "__main__":
    asyncio.run(run_calculation_workflow())
