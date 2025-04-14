import logging
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import KernelPlugin, KernelFunctionFromPrompt

from sk.skills.energy_news_facade import EnergyNewsFacade
from sk.skills.electricity_facade import ElectricityFacade
from sk.skills.weather_facade import WeatherFacade
from sk.orchestrators.semantic_orchestrator import SemanticOrchastrator

class EnergyOrchestrator(SemanticOrchastrator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Energy Orchestrator init")

        energy_news = EnergyNewsFacade()
        electricity = ElectricityFacade()
        weather = WeatherFacade()

        self.kernel = Kernel(
            services=[self.gpt4o_service],
            plugins=[
                KernelPlugin.from_object(plugin_instance=energy_news, plugin_name="energy_news"),
                KernelPlugin.from_object(plugin_instance=electricity, plugin_name="electricity"),
                KernelPlugin.from_object(plugin_instance=weather, plugin_name="weather"),
            ]
        )

    # --------------------------------------------
    # Selection Strategy
    # --------------------------------------------
    def create_selection_strategy(self, agents, default_agent):
        """Speaker selection strategy for the agent group chat."""
        definitions = "\n".join([f"{agent.name}: {agent.description}" for agent in agents])
        selection_function = KernelFunctionFromPrompt(
                function_name="SpeakerSelector",
                prompt_execution_settings=AzureChatPromptExecutionSettings(
                    temperature=0),
                prompt=fr"""
                    You are the next speaker selector. Decide WHAT agent is the best next speaker.

                    - You MUST return ONLY agent name from the list of available agents below.
                    - You MUST return the agent name and nothing else.
                    - The names are case-sensitive and should not be abbreviated or changed.
                    - YOU MUST OBSERVE AGENT USAGE INSTRUCTIONS.

# AVAILABLE AGENTS

{definitions}

# SPECIAL AGENT INVOKATION INSTRUCTIONS
If the user request specifically for electricity production or consumption in Switzerland, only invoke the ElectricityAgent.

# CHAT HISTORY

{{{{$history}}}}
""")

        # Could be lambda. Keeping as function for clarity
        def parse_selection_output(output):
            self.logger.debug(f"Parsing selection: {output}")
            if output.value is not None:
                return output.value[0].content
            return default_agent.name

        return KernelFunctionSelectionStrategy(
                    kernel=self.kernel,
                    function=selection_function,
                    result_parser=parse_selection_output,
                    agent_variable_name="agents",
                    history_variable_name="history")

    # --------------------------------------------
    # Termination Strategy
    # --------------------------------------------
    def create_termination_strategy(self, agents, final_agent, maximum_iterations):
        """
        Create a chat termination strategy that terminates when the final agent is reached.
        params:
            agents: List of agents to trigger termination evaluation
            final_agent: The agent that should trigger termination
            maximum_iterations: Maximum number of iterations before termination
        """
        class CompletionTerminationStrategy(TerminationStrategy):
            async def should_agent_terminate(self, agent, history):
                """Terminate if the last actor is the Responder Agent."""
                logging.getLogger(__name__).debug(history[-1])
                return (agent.name == final_agent.name)

        return CompletionTerminationStrategy(agents=agents,
                                             maximum_iterations=maximum_iterations)

    # --------------------------------------------
    # Create Agent Group Chat
    # --------------------------------------------
    def create_agent_group_chat(self):

        self.logger.debug("Creating Energy chat")

        energy_news_agent = self.create_agent(service_id="gpt-4o",
                                        kernel=self.kernel,
                                        definition_file_path="sk/agents/energy/energy_news.yaml")
        electricity_agent = self.create_agent(service_id="gpt-4o",
                                            kernel=self.kernel,
                                            definition_file_path="sk/agents/energy/electricity.yaml")
        weather_agent = self.create_agent(service_id="gpt-4o",
                                            kernel=self.kernel,
                                            definition_file_path="sk/agents/energy/weather.yaml")
        insights_agent = self.create_agent(service_id="gpt-4o",
                                            kernel=self.kernel,
                                            definition_file_path="sk/agents/energy/insights.yaml")
        responder_agent = self.create_agent(service_id="gpt-4o",
                                            kernel=self.kernel,
                                            definition_file_path="sk/agents/energy/responder.yaml")

        agents=[energy_news_agent, electricity_agent, weather_agent, insights_agent, responder_agent]

        agent_group_chat = AgentGroupChat(
                agents=agents,
                selection_strategy=self.create_selection_strategy(agents, responder_agent),
                termination_strategy = self.create_termination_strategy(
                                         agents=[energy_news_agent, electricity_agent, weather_agent, insights_agent, responder_agent],
                                         final_agent=responder_agent,
                                         maximum_iterations=4))

        return agent_group_chat
