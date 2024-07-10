from typing import Generic

import torch
from tqdm.auto import tqdm

from tinyrl.agent import BaseTorchAgent
from tinyrl.environment import BaseEnvironment
from tinyrl.types import T_Action, T_State


class Reinforce(Generic[T_State, T_Action]):
    def __init__(
        self,
        env: BaseEnvironment[T_State, T_Action],
        agent: BaseTorchAgent[T_State, T_Action],
        optimizer: torch.optim.Optimizer,
        *,
        gamma: float = 0.99,
    ) -> None:
        self._env = env
        self._agent = agent
        self._optimizer = optimizer
        self._gamma = gamma

    def _reinforce(
        self,
        states: list[T_State],
        actions: list[T_Action],
        rewards: list[float],
    ) -> float:
        self._agent.train()
        self._optimizer.zero_grad()

        returns = 0.0
        loss = next(self._agent.parameters()).new_tensor(0.0, requires_grad=False)
        for state, action, reward in reversed(list(zip(states, actions, rewards))):
            returns = reward + self._gamma * returns
            action_prob = self._agent.prob(state, action)
            loss += -action_prob.log() * returns

        loss.backward()  # type: ignore[no-untyped-call]
        self._optimizer.step()
        return float(loss.item())

    def learn(self, max_episodes: int) -> None:
        with tqdm(range(max_episodes)) as pbar:
            for episode in pbar:
                states, actions, rewards = [], [], []

                self._agent.eval()
                with torch.inference_mode():
                    state = self._env.reset()
                    done = False
                    while not done:
                        states.append(state)
                        action, action_prob = self._agent.sample(state)
                        state, reward, done = self._env.step(action)
                        actions.append(action)
                        rewards.append(reward)

                loss = self._reinforce(states, actions, rewards)
                pbar.set_postfix(
                    episode=episode,
                    loss=loss,
                )
