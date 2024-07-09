from typing import Generic, TypeVar

import torch
from tqdm.auto import tqdm

from tinyrl.actor import BaseActor
from tinyrl.env import BaseEnv
from tinyrl.network import BasePolicyNetwork

_T_State = TypeVar("_T_State")
_T_Action = TypeVar("_T_Action")


class Reinforce(Generic[_T_State, _T_Action]):
    def __init__(
        self,
        env: BaseEnv[_T_State, _T_Action],
        actor: BaseActor[_T_Action],
        policy: BasePolicyNetwork[_T_State],
        optimizer: torch.optim.Optimizer,
        gamma: float,
    ) -> None:
        self._env = env
        self._actor = actor
        self._policy = policy
        self._optimizer = optimizer
        self._gamma = gamma

    def _reinforce(
        self,
        states: list[_T_State],
        actions: list[_T_Action],
        rewards: list[float],
    ) -> float:
        self._policy.train()
        self._optimizer.zero_grad()

        returns = 0.0
        loss = next(self._policy.parameters()).new_tensor(0.0)
        for state, action, reward in reversed(list(zip(states, actions, rewards))):
            returns = reward + self._gamma * returns
            action_probs = self._policy(state)
            _, action_prob = self._actor(action_probs)
            loss += -action_prob.log() * returns

        loss.backward()  # type: ignore[no-untyped-call]
        self._optimizer.step()
        return float(loss.item())

    def learn(self, max_episodes: int) -> None:
        with tqdm(range(max_episodes)) as pbar:
            for episode in pbar:
                states, actions, rewards = [], [], []

                self._policy.eval()
                with torch.inference_mode():
                    state = self._env.reset()
                    done = False
                    while not done:
                        states.append(state)
                        action_probs = self._policy(state)
                        action, action_prob = self._actor(action_probs)
                        state, reward, done = self._env.step(action)
                        actions.append(action)
                        rewards.append(reward)

                loss = self._reinforce(states, actions, rewards)
                pbar.set_postfix(
                    episode=episode,
                    loss=loss,
                    reward=sum(rewards),
                )
