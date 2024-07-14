from typing import Generic

import torch
from tqdm.auto import tqdm

from tinyrl.agent import BaseTorchAgent
from tinyrl.environment import BaseEnvironment
from tinyrl.network import BaseValueNetwork
from tinyrl.types import T_Action, T_State


class PPO(Generic[T_State, T_Action]):
    def __init__(
        self,
        env: BaseEnvironment[T_State, T_Action],
        agent: BaseTorchAgent[T_State, T_Action],
        value: BaseValueNetwork[T_State],
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        clip: float = 0.5,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
    ) -> None:
        self._env = env
        self._agent = agent
        self._value = value
        self._policy_optimizer = policy_optimizer
        self._value_optimizer = value_optimizer
        self._gamma = gamma
        self._epsilon = epsilon
        self._clip = clip
        self._value_weight = value_weight
        self._entropy_weight = entropy_weight

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        returns: list[float] = []
        return_ = 0.0
        for reward in reversed(rewards):
            return_ = reward + self._gamma * return_
            returns.insert(0, return_)
        return returns

    def _ppo(
        self,
        states: list[T_State],
        actions: list[T_Action],
        returns: list[float],
        old_log_probs: list[float],
    ) -> float:
        self._agent.train()
        self._value.train()
        self._policy_optimizer.zero_grad()
        self._value_optimizer.zero_grad()

        new_tensor = next(self._agent.parameters()).new_tensor

        _returns = new_tensor(returns)
        _old_log_probs = new_tensor(old_log_probs)

        action_dists = [self._agent.dist(state) for state in states]
        values = torch.concat([self._value(state) for state in states])

        new_log_probs = torch.stack([dist.log_prob(action) for dist, action in zip(action_dists, actions)])

        advantages = _returns - values
        ratio = (new_log_probs - _old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = torch.stack([dist.entropy() for dist in action_dists]).mean()

        loss = policy_loss + self._value_weight * value_loss + self._entropy_weight * entropy_loss

        loss.backward()  # type: ignore[no-untyped-call]
        self._policy_optimizer.step()
        self._value_optimizer.step()

        return float(loss.item())

    def learn(self, max_episodes: int) -> None:
        with tqdm(range(max_episodes)) as pbar:
            for episode in pbar:
                states, actions, rewards, old_log_probs = [], [], [], []

                self._agent.eval()
                self._value.eval()
                with torch.inference_mode():
                    state = self._env.reset()
                    done = False
                    while not done:
                        states.append(state)
                        action, action_prob = self._agent.sample(self._env, state)
                        state, reward, done = self._env.step(action)

                        actions.append(action)
                        rewards.append(reward)
                        old_log_probs.append(float(action_prob.item()))

                returns = self._compute_returns(rewards)
                loss = self._ppo(states, actions, returns, old_log_probs)
                pbar.set_postfix(
                    episode=episode,
                    loss=loss,
                )
