from typing import Generic

from tqdm.auto import tqdm

from tinyrl.agent import QLearningAgent
from tinyrl.environment import BaseEnvironment, ICategoricalActionEnvironment
from tinyrl.types import T_Action, T_State


class QLearning(Generic[T_State, T_Action]):
    def __init__(
        self,
        env: BaseEnvironment[T_State, T_Action],
        agent: QLearningAgent[T_State, T_Action],
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        self._env = env
        self._agent = agent
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

    def update(
        self,
        state: T_State,
        action: T_Action,
        reward: float,
        next_state: T_State,
    ) -> None:
        if isinstance(self._env, ICategoricalActionEnvironment):
            available_actions = self._env.available_actions(state)
        else:
            available_actions = self._agent.actions
        current_q = self._agent.get_q_value(state, action)
        next_q_values = [self._agent.get_q_value(next_state, next_action) for next_action in available_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        new_q = current_q + self._alpha * (reward + self._gamma * max_next_q - current_q)
        self._agent.set_q_value(state, action, new_q)

    def learn(self, max_episodes: int) -> None:
        with tqdm(range(max_episodes)) as pbar:
            for episode in pbar:
                state = self._env.reset()
                done = False
                while not done:
                    action, _ = self._agent.sample(self._env, state)
                    next_state, reward, done = self._env.step(action)
                    self.update(state, action, reward, next_state)
                    state = next_state
                    pbar.set_postfix(reward=reward)
