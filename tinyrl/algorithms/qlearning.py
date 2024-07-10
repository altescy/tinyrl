from typing import Generic

from tqdm.auto import tqdm

from tinyrl.agent import QLearningAgent
from tinyrl.environment import BaseEnvironment
from tinyrl.types import T_Action, T_State


class QLearning(Generic[T_State, T_Action]):
    def __init__(
        self,
        env: BaseEnvironment[T_State, T_Action],
        agent: QLearningAgent[T_State, T_Action],
    ) -> None:
        self._env = env
        self._agent = agent

        self._q_values: dict[tuple[T_State, T_Action], float] = {}

    def learn(self, max_episodes: int) -> None:
        with tqdm(range(max_episodes)) as pbar:
            for episode in pbar:
                state = self._env.reset()
                done = False
                while not done:
                    action, _ = self._agent.sample(state)
                    next_state, reward, done = self._env.step(action)
                    self._agent.update(state, action, reward, next_state)
                    state = next_state
                    pbar.set_postfix(reward=reward)
