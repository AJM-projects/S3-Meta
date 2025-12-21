# S3-Meta: Selective State-Space Meta-Reinforcement Learning

S3-Meta is a meta-reinforcement learning framework that leverages Selective State-Space Models for efficient task inference and adaptation. This repository contains the official implementation used for the experiments in our paper.

## Overview

The core of S3-Meta is the **S3 Encoder**, which uses a State-Space Model architecture to process sequences of observations, actions, and rewards. This allows the agent to maintain a compressed representation of the task and adapt its policy accordingly.

### Key Components:
- **Agents**: Implementations of S3-Meta and various baselines (e.g., Humplik, Belief-based).
- **Environments**: A suite of BAMDP (Bayes-Adaptive Markov Decision Process) environments, including Cheetah, MiniGrid, and PointMass tasks.
- **Models**: The variational S3 encoder and corresponding decoders for reward and task reconstruction.
- **Configs**: Flexible configuration management for different agents and environments.

## Usage

### Training an Agent

The main entry point for training is `main.py`. You can configure the agent and environment within the script or via configuration files.

To train the S3-Meta agent on the `DelayedMAB` environment:
```bash
python main.py
```

### Repository Structure

- `agents/`: Core agent logic and configurations.
- `environments/`: BAMDP environment wrappers and implementations.
- `models/`: Neural network architectures (S3 Encoder, Decoders).
- `configs/`: Hyperparameters and task generators.
- `callbacks/`: Training callbacks for evaluation and logging.
- `utils/`: Helper functions and wrappers for external environments.

