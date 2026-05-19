"""Per-agent Optuna search space definitions."""

from optuna import Trial


def _shared_params(trial: Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "vae_buffer_size": trial.suggest_categorical(
            "vae_buffer_size", [30, 50, 100, 150]
        ),
        "update_every_n": trial.suggest_categorical(
            "update_every_n", [1000, 2000, 4000]
        ),
    }


def suggest_s3_meta(trial: Trial) -> dict:
    params = _shared_params(trial)
    params.update(
        {
            "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
            "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "expand": trial.suggest_categorical("expand", [1, 2, 4]),
            "kl_weight": trial.suggest_float("kl_weight", 0.01, 1.0, log=True),
            "reward_weight": trial.suggest_float("reward_weight", 0.5, 2.0),
            "contrastive_weight": trial.suggest_float("contrastive_weight", 0.1, 2.0),
        }
    )
    return params


def suggest_belief(trial: Trial) -> dict:
    params = _shared_params(trial)
    params.update(
        {
            "encoder_gru_hidden_size": trial.suggest_categorical(
                "encoder_gru_hidden_size", [32, 64, 128, 256]
            ),
            "num_gru_layers": trial.suggest_int("num_gru_layers", 1, 3),
            "kl_weight": trial.suggest_float("kl_weight", 0.01, 1.0, log=True),
            "task_weight": trial.suggest_float("task_weight", 0.5, 2.0),
        }
    )
    return params


def suggest_humplik(trial: Trial) -> dict:
    # Humplik uses LSTM but shares the same config parameter names as GRU
    params = _shared_params(trial)
    params.update(
        {
            "encoder_gru_hidden_size": trial.suggest_categorical(
                "encoder_gru_hidden_size", [32, 64, 128, 256]
            ),
            "num_gru_layers": trial.suggest_int("num_gru_layers", 1, 3),
            "kl_weight": trial.suggest_float("kl_weight", 0.01, 1.0, log=True),
            "task_weight": trial.suggest_float("task_weight", 0.5, 2.0),
        }
    )
    return params


def suggest_splagger(trial: Trial) -> dict:
    # Paper (Appendix B) tunes lr over [3e-5, 3e-3]; other shared params follow
    # the same sweep as other agents. Architecture: GRU hidden size and depth.
    # No kl/task/contrastive weights — SplAggerAgent asserts those are off.
    params = _shared_params(trial)
    # Override lr to match the paper's wider range [3e-5, 3e-3]
    params["lr"] = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    params.update(
        {
            "encoder_gru_hidden_size": trial.suggest_categorical(
                "encoder_gru_hidden_size", [64, 128, 256, 512]
            ),
            "num_gru_layers": trial.suggest_int("num_gru_layers", 1, 3),
            "reward_weight": trial.suggest_float("reward_weight", 0.5, 2.0),
        }
    )
    return params


AGENT_SEARCH_SPACES = {
    "S3_Meta": suggest_s3_meta,
    "belief": suggest_belief,
    "humplik": suggest_humplik,
    "SplAgger": suggest_splagger,
}
