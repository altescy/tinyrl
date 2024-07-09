from __future__ import annotations

import dataclasses
import enum
import time
from typing import Callable, cast

import torch

from tinyrl.actor import BaseActor
from tinyrl.env import BaseEnv
from tinyrl.network import BasePolicyNetwork


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


class GridWorld(BaseEnv["State", "Action"]):
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
        actor: Callable[[State], Action],
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
            action = actor(state)
            state, _, done = self.step(action)
            time.sleep(interval)
            step += 1
        print(f"[ step: {step} ]")
        self.render()


class GridWorldPolicyNetwork(BasePolicyNetwork[State]):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(input_size, output_size)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._linear.weight.new_tensor(
            [state.position.x, state.position.y],
            requires_grad=False,
        ).float()
        return cast(
            torch.Tensor,
            self._linear(state_tensor).softmax(-1),
        )


class GridWorldActor(BaseActor[Action]):
    def __call__(self, probs: torch.Tensor) -> tuple[Action, torch.Tensor]:
        action = int(torch.multinomial(probs, 1).item())
        return Action(action), probs[action]


def run() -> None:
    from tinyrl.reinforce import Reinforce

    torch.manual_seed(16)

    env = GridWorld(5)
    actor = GridWorldActor()
    policy = GridWorldPolicyNetwork(2, 4)
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
        env.animate(lambda state: actor(policy(state))[0])


if __name__ == "__main__":
    run()
