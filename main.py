import gc
from policy_network import PPOWithInfo
from callbacks.agent_callbacks import create_agent_callbacks
from environments.environment_builder import build_environment
from agents import build_agent
from agents.agent_configs import get_config


def train_agent(
    env_name: str | tuple[str, str],
    agent_name: str,
    num_experiment: int = 1,
    verbose=0,
    main_log_dir="trained_agents",
):
    for experiment in range(num_experiment):
        config = get_config(
            agent_name,
            env_name if isinstance(env_name, str) else f"{env_name[0]}_{env_name[1]}",
        )
        agent = build_agent(agent_name, config)
        environment = build_environment(
            config.env_register, agent, config, config.train_tasks
        )()
        test_environment = build_environment(
            config.env_register, agent, config, config.test_tasks
        )()

        model = PPOWithInfo(
            "MlpPolicy",
            environment,
            verbose=verbose,
            tensorboard_log=f"{main_log_dir}/{config.env_name}/{agent_name}",
            n_steps=config.update_every_n,
        )

        print(f"tensorboard --logdir {model.tensorboard_log}")
        callbacks = create_agent_callbacks(agent, config, test_environment)
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
        gc.collect()

if __name__ == "__main__":
    train_agent(env_name="CheetahDir", agent_name="S3_Meta", num_experiment=1)
