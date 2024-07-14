from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias, cast

import numpy
import torch

from tinyrl.agent import BaseTorchAgent
from tinyrl.distribution import TorchCategoricalDistribution
from tinyrl.environment import BaseEnvironment
from tinyrl.network import BasePolicyNetwork

State: TypeAlias = Literal[0]
Action: TypeAlias = int


class Bandit(BaseEnvironment[State, Action]):
    def __init__(self, num_bandints: int) -> None:
        self._num_bandits = num_bandints
        self._probs = numpy.random.rand(num_bandints)

    def reset(self) -> State:
        return 0

    def step(self, action: Action) -> tuple[State, float, bool]:
        reward = 1 if numpy.random.rand() < self._probs[action] else -1
        return 0, reward, True

    def available_actions(self, state: State) -> set[Action]:
        return set(range(self._num_bandits))

    def evaluate(
        self,
        act: Callable[[Bandit, State], Action],
        *,
        num_trials: int = 100,
    ) -> None:
        reward = 0.0
        for _ in range(num_trials):
            reward += self.step(act(self, 0))[1]

        random_reward = num_trials * numpy.sum(2 * self._probs - 1) / self._num_bandits

        print(f"Random reward: {random_reward:.2f}")
        print(f"Actual reward: {reward:.2f}")


class BanditPolicyNetwork(BasePolicyNetwork[State]):
    def __init__(self, num_bandits: int) -> None:
        super().__init__()
        self._layer = torch.nn.Linear(1, num_bandits)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._layer.weight.new_tensor([state], requires_grad=False)
        return cast(torch.Tensor, self._layer(state_tensor).softmax(-1))


class BanditAgent(BaseTorchAgent[State, Action]):
    def __init__(self, policy: BanditPolicyNetwork) -> None:
        super().__init__()
        self._policy = policy

    def dist(
        self,
        state: State,
        *,
        available_actions: set[Action] | None = None,
    ) -> TorchCategoricalDistribution:
        del available_actions
        probs = self._policy(state)
        actions = list(range(len(probs)))
        return TorchCategoricalDistribution(probs, actions)


def run_reinforce() -> None:
    from tinyrl.algorithms import Reinforce

    numpy.random.seed(16)
    torch.manual_seed(16)

    num_bandits = 10

    env = Bandit(num_bandits)
    policy = BanditPolicyNetwork(num_bandits)
    agent = BanditAgent(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    reinforce = Reinforce(env, agent, optimizer)
    reinforce.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.evaluate(agent.act)


def run_qlearning() -> None:
    from tinyrl.agent import QLearningAgent
    from tinyrl.algorithms import QLearning

    numpy.random.seed(16)

    num_bandits = 10
    actions: set[Action] = set(range(num_bandits))

    env = Bandit(num_bandits)
    agent = QLearningAgent[State, Action](actions)
    qlearning = QLearning(env, agent)

    qlearning.learn(max_episodes=1000)

    env.evaluate(agent.act)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [reinforce|qlearning]")
        sys.exit(1)

    if sys.argv[1] == "reinforce":
        run_reinforce()
    elif sys.argv[1] == "qlearning":
        run_qlearning()
    else:
        print(f"Invalid algorithm: {sys.argv[1]}")
        sys.exit(1)
