# Example taken from https://github.com/run-llama/llama-agents/blob/main/examples/agentic_server.py
# Used for deep dive into code
# https://youtu.be/_dE3beiGeyk
import dotenv

dotenv.load_dotenv()
from llama_agents import (
    AgentService,
    AgentOrchestrator,
    ControlPlaneServer,
    SimpleMessageQueue,
)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import logging

# change logging level to enable or disable more verbose logging
logging.getLogger("llama_agents").setLevel(logging.INFO)

# create our multi-agent framework components
message_queue = SimpleMessageQueue()
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=OpenAI()),
)


# create a tool
def get_the_secret_fact() -> str:
    """Returns the secret fact."""
    return "The secret fact is: A baby llama is called a 'Cria'."


tool = FunctionTool.from_defaults(fn=get_the_secret_fact)


# create our agents
worker1 = FunctionCallingAgentWorker.from_tools([tool], llm=OpenAI())
worker2 = FunctionCallingAgentWorker.from_tools([], llm=OpenAI())
agent1 = worker1.as_agent()
agent2 = worker2.as_agent()

agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the secret fact.",
    service_name="secret_fact_agent",
    host="localhost",
    port=8003
)
agent_server_2 = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for getting random dumb facts.",
    service_name="dumb_fact_agent",
    host="localhost",
    port=8004
)

from llama_agents import ServerLauncher, CallableMessageConsumer


# Additional human consumer
def handle_result(message) -> None:
    print(f"Got result:", message.data)


# the final result is published to a "human" consumer
# so we define one to handle it!
human_consumer = CallableMessageConsumer(
    handler=handle_result, message_type="human"
)

# Define Launcher
launcher = ServerLauncher(
    [agent_server_1, agent_server_2],
    control_plane,
    message_queue,
    additional_consumers=[human_consumer]
)

launcher.launch_servers()
