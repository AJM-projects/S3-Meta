from .S3_agent import S3Agent
from .belief_agent import BeliefAgent
from .humplik_agent import HumplikAgent


def build_agent(agent_name: str, config):
    """
    Factory function to build agents based on agent name.

    Args:
        agent_name: Name of the agent to build
        config: Configuration namespace

    Returns:
        Agent instance
    """
    agent_mapping = {
        "S3_Meta": S3Agent,
        "belief": BeliefAgent,
        "humplik": HumplikAgent,
    }

    if agent_name not in agent_mapping:
        raise ValueError(f"Unknown agent: {agent_name}")

    return agent_mapping[agent_name](config)
