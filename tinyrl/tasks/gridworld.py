from __future__ import annotations

import dataclasses
import enum
import time
from typing import Callable, cast

import torch

from tinyrl.actor import BaseActor
from tinyrl.environment import BaseEnvironment
from tinyrl.network import BasePolicyNetwork, BaseValueNetwork


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
        act: Callable[[State], Action],
        *,
        max_steps: int | None = None,
        interval: float = 0.1,
    ) -> None:
        step = 0
        done = False
        state = self.reset()
        while not done and (max_steps is None or step < max_steps):
            print(f"[ step: {step} ]")
            self.render()
            action = act(state)
            state, _, done = self.step(action)
            time.sleep(interval)
            step += 1
        print(f"[ step: {step} ]")
        self.render()


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


class GridWorldActor(BaseActor[Action]):
    def select(self, probs: torch.Tensor, action: Action) -> torch.Tensor:
        return probs[action]

    def sample(self, probs: torch.Tensor) -> tuple[Action, float]:
        action = int(torch.multinomial(probs, 1).item())
        return Action(action), float(probs[action].item())


def run_reinforce() -> None:
    from tinyrl.reinforce import Reinforce

    torch.manual_seed(16)

    env = GridWorld(5)
    actor = GridWorldActor()
    policy = GridWorldPolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    reinforce = Reinforce(
        env=env,
        actor=actor,
        policy=policy,
        optimizer=optimizer,
        gamma=0.99,
    )

    reinforce.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.animate(lambda state: actor.sample(policy(state))[0])


def run_ppo() -> None:
    from tinyrl.ppo import PPO

    torch.manual_seed(16)

    env = GridWorld(5)
    actor = GridWorldActor()
    policy = GridWorldPolicyNetwork()
    value = GridWorldValueNetwork()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.05)

    ppo = PPO(
        env=env,
        actor=actor,
        policy=policy,
        value=value,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )

    ppo.learn(max_episodes=1000)

    policy.eval()
    with torch.inference_mode():
        env.animate(lambda state: actor.sample(policy(state))[0])


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [reinforce|ppo]")
        sys.exit(1)

    if sys.argv[1] == "reinforce":
        run_reinforce()
    elif sys.argv[1] == "ppo":
        run_ppo()
    else:
        print(f"Invalid algorithm: {sys.argv[1]}")
        sys.exit(1)
