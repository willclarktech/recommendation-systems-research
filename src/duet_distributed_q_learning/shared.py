import asyncio

import gym
from torch import Tensor

# (current sum, dealer card, usable ace)
BlackjackObservation = tuple[int, int, bool]

N_EPOCHS = 5
NETWORK_URL = "http://0.0.0.0:5000/"
OWNER_TAG = "own"
SCIENTIST_TAG = "sci"


class QLearningAgent:
    def __init__(self, initial_q: Tensor, alpha: float, gamma: float) -> None:
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.q_table = initial_q.detach().clone()

    def act(self, observation: BlackjackObservation) -> int:
        return int(self.get_q_values_for_observation(observation).argmax())

    def get_q_values_for_observation(self, observation: BlackjackObservation) -> Tensor:
        current_sum, dealer_card, usable_ace = observation
        return self.q_table[current_sum][dealer_card][int(usable_ace)]

    def update(
        self,
        observation: BlackjackObservation,
        action: int,
        reward: float,
        observation_next: BlackjackObservation,
    ) -> None:
        q_values = self.get_q_values_for_observation(observation)
        max_next_q_value = self.get_q_values_for_observation(observation_next).max()
        q_values[action] = q_values[action] + self.alpha * (
            reward + self.gamma * max_next_q_value - q_values[action]
        )


def run_iteration(agent: QLearningAgent, environment: gym.Env, train: bool) -> float:
    ret = 0.0
    observation = environment.reset()
    done = False

    while not done:
        action = agent.act(observation)
        observation_next, reward, done, _ = environment.step(action)
        ret += reward
        if train:
            agent.update(observation, action, reward, observation_next)
        observation = observation_next

    return ret


def run_epoch(
    n_iterations: int, initial_q: Tensor, train: bool = True
) -> tuple[Tensor, list[float]]:
    agent = QLearningAgent(initial_q, alpha=0.01, gamma=0.01)
    environment = gym.make("Blackjack-v0")
    rets = []

    for _ in range(n_iterations):
        ret = run_iteration(agent, environment, train)
        rets.append(ret)

    return agent.q_table, rets


async def wait_for_remote_object(duet, tag: str, poll_interval_secs: float = 1.0):
    while True:
        try:
            return duet.store[tag]
        except KeyError:
            await asyncio.sleep(poll_interval_secs)
