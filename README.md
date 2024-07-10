# TinyRL: Minimal Reinforcement Learning Implementations

This repository contains minimal implementations of various reinforcement
learning algorithms with NumPy and PyTorch.

## Algorithms

- [x] [Q-Learning](./tinyrl/algorithms/qlearning.py)
- [ ] Deep Q-Network (DQN)
- [x] [REINFORCE](./tinyrl/algorithms/reinforce.py)
- [ ] Actor-Critic
- [x] [Proximal Policy Optimization (PPO)](./tinyrl/algorithms/ppo.py)

## Demos

- [x] [Bandit](./tinyrl/tasks/bandit.py)
- [x] [GridWorld](./tinyrl/tasks/gridworld.py)

## Usage

- Setup the environment:

```shell
git clone https://github.com/altescy/tinyrl.git
cd tinyrl
poetry install
```

- Run the example:

```shell
poetry run python -m tinyrl.tasks.gridworld ppo
```
