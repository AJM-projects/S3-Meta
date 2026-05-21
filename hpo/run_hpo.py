"""CLI entry point for hyperparameter optimisation.

Usage:
    python -m hpo.run_hpo --agent S3_Meta --env MiniGridKeyDoor
    python -m hpo.run_hpo --agent S3_Meta --env TemporalMaze
    python -m hpo.run_hpo --agent S3_Meta --env MiniGridTwoGoal
    python -m hpo.run_hpo --agent belief --env MAB --n_trials 30 --timesteps_fraction 0.2
    python -m hpo.run_hpo --agent humplik --env DelayedMAB --storage sqlite:///hpo.db

    # Export results from an existing study without running new trials:
    python -m hpo.run_hpo --agent S3_Meta --env CheetahDir --storage sqlite:///cheetahdir_hpo.db --export_only
"""

import argparse
import json
import os
from datetime import datetime

import optuna

from configs.environment_configs import ENVIRONMENT_CONFIGS
from hpo.objective import make_objective
from hpo.search_spaces import AGENT_SEARCH_SPACES


def _results_path(agent_name: str, env_name: str) -> str:
    output_dir = os.path.join("hpo_results", env_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{agent_name}.json")


def export_study_results(
    study: optuna.Study,
    output_path: str,
    agent_name: str,
    env_name: str,
) -> None:
    """Save all trial results and the best config to a JSON file."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        print("No completed trials to export.")
        return

    best = study.best_trial
    results = {
        "study_name": study.study_name,
        "agent": agent_name,
        "env": env_name,
        "exported_at": datetime.now().isoformat(),
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
        "best": {
            "trial_number": best.number,
            "value": best.value,
            "params": best.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in sorted(study.trials, key=lambda t: t.number)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO for S3-Meta agents")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(AGENT_SEARCH_SPACES),
        help="Agent to optimise",
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=list(ENVIRONMENT_CONFIGS),
        help="Environment key from ENVIRONMENT_CONFIGS",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timesteps_fraction",
        type=float,
        default=0.3,
        help="Fraction of full total_timesteps to use per trial (default 0.3)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10_000,
        help="PPO steps between evaluations / Optuna reports (default 10000)",
    )
    parser.add_argument(
        "--study_name",
        default=None,
        help="Optuna study name (defaults to <agent>_<env>_hpo)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. sqlite:///hpo.db",
    )
    parser.add_argument(
        "--log_dir",
        default="hpo_logs",
        help="Root directory for per-trial tensorboard logs (default hpo_logs)",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Trials before pruning activates (default 5)",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=3,
        help="Evaluations before a trial can be pruned (default 3)",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Skip optimisation; load existing study from --storage and export results to JSON",
    )
    args = parser.parse_args()

    study_name = args.study_name or f"{args.agent}_{args.env}_hpo"

    if args.export_only:
        if not args.storage:
            parser.error(
                "--export_only requires --storage to point to an existing study DB"
            )
        study = optuna.load_study(study_name=study_name, storage=args.storage)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=args.storage,
            load_if_exists=True,
        )
        objective = make_objective(
            agent_name=args.agent,
            env_name=args.env,
            timesteps_fraction=args.timesteps_fraction,
            eval_freq=args.eval_freq,
            log_dir=args.log_dir,
        )
        study.optimize(objective, n_trials=args.n_trials)

    output_path = _results_path(args.agent, args.env)
    export_study_results(study, output_path, args.agent, args.env)

    best = study.best_trial
    print(f"\n{'=' * 50}")
    print(f"Study:      {study_name}")
    print(f"Best trial: #{best.number}")
    print(f"Best value: {best.value:.4f}")
    print("Best params:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
