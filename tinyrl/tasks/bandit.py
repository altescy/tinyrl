from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias, cast

import numpy
import torch

from tinyrl.actor import BaseActor
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

    def evaluate(
        self,
        sampler: Callable[[], Action],
        *,
        num_trials: int = 100,
    ) -> None:
        reward = 0.0
        for _ in range(num_trials):
            reward += self.step(sampler())[1]

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


class BanditActor(BaseActor[Action]):
    def __init__(self, num_bandits: int) -> None:
        self._num_bandits = num_bandits

    def __call__(self, probs: torch.Tensor) -> tuple[Action, torch.Tensor]:
        m = torch.distributions.Categorical(probs)  # type: ignore[no-untyped-call]
        action = int(m.sample().item())  # type: ignore[no-untyped-call]
        return action, probs[action]


def run() -> None:
    from tinyrl.reinforce import Reinforce

    numpy.random.seed(16)
    torch.manual_seed(16)

    num_bandits = 10

    env = Bandit(num_bandits)
    actor = BanditActor(num_bandits)
    policy = BanditPolicyNetwork(num_bandits)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    reinforce = Reinforce(env, actor, policy, optimizer, gamma=0.99)
    reinforce.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.evaluate(lambda: actor(policy(0))[0])


if __name__ == "__main__":
    run()
