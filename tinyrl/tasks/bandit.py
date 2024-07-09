from __future__ import annotations

from typing import Literal, TypeAlias, cast

import numpy
import torch

from tinyrl.actor import BaseActor
from tinyrl.env import BaseEnv
from tinyrl.network import BasePolicyNetwork

State: TypeAlias = Literal[0]
Action: TypeAlias = int


class Bandit(BaseEnv[State, Action]):
    def __init__(self, num_bandints: int) -> None:
        self._num_bandits = num_bandints
        self._probs = numpy.random.rand(num_bandints)

    def reset(self) -> State:
        return 0

    def step(self, action: Action) -> tuple[State, float, bool]:
        reward = 1 if numpy.random.rand() < self._probs[action] else -1
        return 0, reward, True


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

    numpy.random.seed(0)
    torch.manual_seed(16)

    env = Bandit(10)
    actor = BanditActor(10)
    policy = BanditPolicyNetwork(10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    reinforce = Reinforce(env, actor, policy, optimizer, gamma=0.99)
    reinforce.learn(max_episodes=1000)

    num_trials = 100
    policy.eval()
    with torch.inference_mode():
        reward = 0.0
        for _ in range(num_trials):
            reward += env.step(actor(policy(0))[0])[1]

    print(f"Earned reward in {num_trials} trials: {reward}")


if __name__ == "__main__":
    run()
