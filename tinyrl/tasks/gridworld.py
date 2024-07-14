from __future__ import annotations

import dataclasses
import enum
import time
from typing import Callable, cast

import numpy
import torch

from tinyrl.agents import BaseTorchAgent
from tinyrl.constraints import CategoricalActionConstraint
from tinyrl.distributions import TorchCategoricalDistribution
from tinyrl.environments import BaseEnvironment
from tinyrl.networks import BasePolicyNetwork, BaseValueNetwork


@dataclasses.dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclasses.dataclass(frozen=True)
class State:
    position: Position
    done: bool = False


class Action(int, enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld(BaseEnvironment["State", "Action"]):
    def __init__(self, size: int) -> None:
        self._size = size
        self._state = State(Position(0, 0))
        self._goal = Position(size - 1, size - 1)

    def reset(self) -> State:
        self._state = State(Position(0, 0))
        return self._state

    def step(self, action: Action) -> tuple[State, float, bool]:
        position = self._state.position
        match action:
            case Action.UP:
                position = dataclasses.replace(self._state.position, y=max(0, self._state.position.y - 1))
            case Action.DOWN:
                position = dataclasses.replace(self._state.position, y=min(self._size - 1, self._state.position.y + 1))
            case Action.LEFT:
                position = dataclasses.replace(self._state.position, x=max(0, self._state.position.x - 1))
            case Action.RIGHT:
                position = dataclasses.replace(self._state.position, x=min(self._size - 1, self._state.position.x + 1))
            case _:
                raise ValueError("Invalid action")

        self._state = dataclasses.replace(
            self._state,
            position=position,
        )

        done = self._state.position == self._goal
        reward = 0 if done else -1
        return self._state, reward, done

    def available_actions(self, state: State) -> set[Action]:
        actions = set(Action)
        if state.position.x == 0:
            actions.discard(Action.LEFT)
        if state.position.x == self._size - 1:
            actions.discard(Action.RIGHT)
        if state.position.y == 0:
            actions.discard(Action.UP)
        if state.position.y == self._size - 1:
            actions.discard(Action.DOWN)
        return actions

    def render(self) -> None:
        for y in range(self._size):
            for x in range(self._size):
                if self._state.position == Position(x, y):
                    print("A", end=" ")
                elif self._goal == Position(x, y):
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()

    def animate(
        self,
        act: Callable[[GridWorld, State], Action],
        *,
        max_steps: int | None = None,
        interval: float = 0.1,
    ) -> None:
        step = 0
        done = False
        state = self.reset()
        while not done and (max_steps is None or step < max_steps):
            self.render()
            print(f"step: {step}")
            print("\033[F" * (self._size + 1), end="")
            action = act(self, state)
            state, _, done = self.step(action)
            time.sleep(interval)
            step += 1
        self.render()
        print(f"step: {step}")


class GridWorldPolicyNetwork(BasePolicyNetwork[State]):
    def __init__(self) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(2, 4)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._linear.weight.new_tensor(
            [state.position.x, state.position.y],
            requires_grad=False,
        ).float()
        return cast(
            torch.Tensor,
            self._linear(state_tensor).softmax(-1),
        )


class GridWorldValueNetwork(BaseValueNetwork[State]):
    def __init__(self) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(2, 1)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._linear.weight.new_tensor(
            [state.position.x, state.position.y],
            requires_grad=False,
        ).float()
        return cast(torch.Tensor, self._linear(state_tensor))


class GridWorldAgent(BaseTorchAgent[State, Action]):
    def __init__(self, policy: GridWorldPolicyNetwork) -> None:
        super().__init__(CategoricalActionConstraint())
        self._policy = policy

    def dist(self, state: State) -> TorchCategoricalDistribution:
        action_probs = self._policy(state)
        actions = [Action(i) for i in range(len(action_probs))]
        return TorchCategoricalDistribution(action_probs, actions)


def run_reinforce() -> None:
    from tinyrl.algorithms import Reinforce

    numpy.random.seed(16)
    torch.manual_seed(16)

    env = GridWorld(5)
    policy = GridWorldPolicyNetwork()
    agent = GridWorldAgent(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    reinforce = Reinforce(
        env=env,
        agent=agent,
        optimizer=optimizer,
    )

    reinforce.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.animate(agent.act)


def run_ppo() -> None:
    from tinyrl.algorithms import PPO

    numpy.random.seed(16)
    torch.manual_seed(16)

    env = GridWorld(5)
    policy = GridWorldPolicyNetwork()
    value = GridWorldValueNetwork()
    agent = GridWorldAgent(policy)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.05)

    ppo = PPO(
        env=env,
        agent=agent,
        value=value,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )

    ppo.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.animate(agent.act)


def run_qlearning() -> None:
    from tinyrl.agents import QLearningAgent
    from tinyrl.algorithms import QLearning

    numpy.random.seed(16)

    env = GridWorld(5)
    agent = QLearningAgent[State, Action](
        set(Action),
        constraint=CategoricalActionConstraint(),
    )
    qlearning = QLearning[State, Action](env=env, agent=agent)

    qlearning.learn(max_episodes=1000)

    env.animate(agent.act)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [ppo|qlearning|reinforce]")
        sys.exit(1)
    if sys.argv[1] == "ppo":
        run_ppo()
    elif sys.argv[1] == "qlearning":
        run_qlearning()
    elif sys.argv[1] == "reinforce":
        run_reinforce()
    else:
        print(f"Invalid algorithm: {sys.argv[1]}")
        sys.exit(1)
