"""Configuration loading utilities."""

from argparse import Namespace
from .base_config import base_config
from .agent_configs import AGENT_CONFIGS
from .environment_configs import ENVIRONMENT_CONFIGS


def get_config(agent_name: str, env_name: str) -> Namespace:
    """
    Create a configuration by merging base config, agent config, and environment config.
    
    Args:
        agent_name: Name of the agent configuration to use
        env_name: Name of the environment configuration to use
        
    Returns:
        Namespace object with merged configuration
    """
    updated_dict = {
        **base_config,
        **AGENT_CONFIGS[agent_name],
        **ENVIRONMENT_CONFIGS[env_name],
        "agent_name": agent_name,
    }
    return Namespace(**updated_dict)