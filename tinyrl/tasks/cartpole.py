from __future__ import annotations

import enum
import time
from typing import Callable, NamedTuple, cast

import numpy
import torch

from tinyrl.agent import BaseTorchAgent
from tinyrl.distribution import TorchCategoricalDistribution
from tinyrl.environment import BaseEnvironment
from tinyrl.network import BasePolicyNetwork, BaseValueNetwork


class Action(int, enum.Enum):
    LEFT = 0
    RIGHT = 1


class State(NamedTuple):
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0


class CartPole(BaseEnvironment[State, Action]):
    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        timestep: float = 0.02,
        theta_threshold: float = 30.0,
        x_threshold: float = 2.4,
        max_steps: int = 1000,
    ) -> None:
        self._gravity = gravity
        self._masspole = masspole
        self._total_mass = masscart + masspole
        self._length = length
        self._polemass_length = masspole * length
        self._force_mag = force_mag
        self._timestep = timestep
        self._theta_threshold_radians = theta_threshold * 2 * numpy.pi / 360
        self._x_threshold = x_threshold
        self._max_steps = max_steps

        self._state: State
        self._total_steps: int
        self.reset()

    def reset(self) -> State:
        self._state = State(
            float(numpy.random.uniform(low=-0.05, high=0.05)),
            float(numpy.random.uniform(low=-0.05, high=0.05)),
            float(numpy.random.uniform(low=-0.05, high=0.05)),
            float(numpy.random.uniform(low=-0.05, high=0.05)),
        )
        self._total_steps = 0
        return self._state

    def step(self, action: int) -> tuple[State, float, bool]:
        state = self._state
        x, x_dot, theta, theta_dot = state
        force = self._force_mag if action == 1 else -self._force_mag

        costheta = numpy.cos(theta)
        sintheta = numpy.sin(theta)

        temp = (force + self._polemass_length * theta_dot**2 * sintheta) / self._total_mass
        theta_acc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._masspole * costheta**2 / self._total_mass)
        )
        x_acc = temp - self._polemass_length * theta_acc * costheta / self._total_mass

        x = float(x + self._timestep * x_dot)
        x_dot = float(x_dot + self._timestep * x_acc)
        theta = float(theta + self._timestep * theta_dot)
        theta_dot = float(theta_dot + self._timestep * theta_acc)

        self._state = State(x, x_dot, theta, theta_dot)
        self._total_steps += 1

        done = bool(
            x < -self._x_threshold
            or x > self._x_threshold
            or theta < -self._theta_threshold_radians
            or theta > self._theta_threshold_radians
        )

        if self._total_steps < self._max_steps:
            reward = 0 if not done else -1
        else:
            reward = 0
            done = True

        return self._state, reward, done

    def render(
        self,
        state: State,
        screen_width: int = 40,
        screen_height: int = 10,
    ) -> None:
        x, _, theta, _ = state
        cart_pos = round((x + self._x_threshold) / (2 * self._x_threshold) * screen_width)
        pole_theta = theta

        pole_length = screen_height
        pole_height_step = round(numpy.cos(pole_theta) * pole_length)
        for i in reversed(range(pole_length)):
            pole_x = cart_pos
            pole_theta_step = round(numpy.tan(pole_theta) * i * 2)
            if 0 <= pole_x + pole_theta_step < screen_width and i <= pole_height_step:
                print(" " * (pole_x + pole_theta_step) + "|")
            else:
                print()
        cart = ["-"] * (screen_width + 1)
        cart[cart_pos] = "C"
        print("".join(cart))

    def animate(
        self,
        act: Callable[[CartPole, State], Action],
        *,
        interval: float | None = None,
        screen_width: int = 40,
        screen_height: int = 10,
    ) -> None:
        step = 0
        done = False
        state = self.reset()
        action: Action | None = None
        while not done:
            self.render(state, screen_width, screen_height)
            print(f"step: {step}")
            print("\033[K\033[A" * (screen_height + 2), end="\033[K")
            action = act(self, state)
            state, reward, done = self.step(action)
            if done:
                break
            time.sleep(interval or self._timestep)
            step += 1

        self.render(state, screen_width, screen_height)
        print(f"step: {step}")


class CartPolePolicyNetwork(BasePolicyNetwork[State]):
    def __init__(self) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(4, 2)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._linear.weight.new_tensor(
            state,
            requires_grad=False,
        ).float()
        return cast(
            torch.Tensor,
            self._linear(state_tensor).softmax(-1),
        )


class CartPoleValueNetwork(BaseValueNetwork[State]):
    def __init__(self) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(4, 1)

    def forward(self, state: State) -> torch.Tensor:
        state_tensor = self._linear.weight.new_tensor(
            state,
            requires_grad=False,
        ).float()
        return cast(torch.Tensor, self._linear(state_tensor))


class CartPoleAgent(BaseTorchAgent[State, Action]):
    def __init__(self, policy: CartPolePolicyNetwork) -> None:
        super().__init__()
        self._policy = policy

    def dist(self, state: State) -> TorchCategoricalDistribution:
        action_probs = self._policy(state)
        actions = [Action(i) for i in range(len(action_probs))]
        return TorchCategoricalDistribution(action_probs, actions)


def run_reinforce() -> None:
    from tinyrl.algorithms import Reinforce

    numpy.random.seed(16)
    torch.manual_seed(16)

    env = CartPole()
    policy = CartPolePolicyNetwork()
    agent = CartPoleAgent(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    reinforce = Reinforce(
        env=env,
        agent=agent,
        optimizer=optimizer,
    )

    reinforce.learn(max_episodes=2000)

    policy.eval()
    with torch.inference_mode():
        env.animate(agent.act)


def run_ppo() -> None:
    from tinyrl.algorithms import PPO

    numpy.random.seed(16)
    torch.manual_seed(16)

    env = CartPole()
    policy = CartPolePolicyNetwork()
    value = CartPoleValueNetwork()
    agent = CartPoleAgent(policy)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.01)

    ppo = PPO(
        env=env,
        agent=agent,
        value=value,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )

    ppo.learn(max_episodes=2000)

    policy.eval()
    with torch.inference_mode():
        env.animate(agent.act)


def run_qlearning() -> None:
    from tinyrl.agent import QLearningAgent
    from tinyrl.algorithms import QLearning

    numpy.random.seed(16)

    env = CartPole()
    agent = QLearningAgent[State, Action](set(Action))
    qlearning = QLearning(env=env, agent=agent)

    qlearning.learn(max_episodes=2000)

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
