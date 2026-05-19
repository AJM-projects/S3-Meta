"""Agent-specific configurations."""

S3_config = {
    "decode_reward": True,
    "decode_task": False,
    "use_decoder": True,
    "contrastive_task_loss": True,
    "num_layers": 2,
    "d_model": 32,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "include_conv": False,
    "layer_norm": True,
    "residual": True,
}

belief_config = {
    "decode_reward": False,
    "decode_task": True,
    "use_decoder": True,
}

humplik_config = {
    "decode_reward": False,
    "decode_task": True,
    "use_decoder": True,
    "contrastive_task_loss": False,
    "use_kl_loss": True,
}

splagger_config = {
    "decode_reward": True,
    "decode_task": False,
    "use_decoder": True,
    "contrastive_task_loss": False,
    "use_kl_loss": False,
}

AGENT_CONFIGS = {
    "S3_Meta": S3_config,
    "belief": belief_config,
    "humplik": humplik_config,
    "SplAgger": splagger_config,
}
