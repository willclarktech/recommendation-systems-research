import random

import gym  # type: ignore
import numpy as np

# (current sum, dealer card, usable ace)
BlackjackObservation = tuple[int, int, bool]


class BlackjackAgent:
    def __init__(self) -> None:
        self.name = "null"

    def act(self, _observation: BlackjackObservation) -> int:
        return 0

    def update(
        self,
        _observation: BlackjackObservation,
        _action: int,
        _reward: float,
        _observation_next: BlackjackObservation,
    ) -> None:
        pass


class RandomAgent(BlackjackAgent):
    def __init__(self, environment: gym.Env) -> None:
        super().__init__()
        self.name = "random"
        self.action_space_size = environment.action_space.n

    def act(self, _observation: BlackjackObservation) -> int:
        return random.randrange(self.action_space_size)


class Stick17Agent(BlackjackAgent):
    def __init__(self) -> None:
        super().__init__()
        self.name = "stick17"

    def act(self, observation: BlackjackObservation) -> int:
        current_sum = observation[0]
        # stick at 17 or higher
        return not current_sum >= 17


class QLearningAgent(BlackjackAgent):
    def __init__(self, environment: gym.Env, alpha: float, gamma: float) -> None:
        super().__init__()
        self.name = "q-learning"
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        (
            current_sum_space,
            dealer_card_space,
            usable_ace_space,
        ) = environment.observation_space.spaces
        self.q_table = np.zeros(
            (
                current_sum_space.n,
                dealer_card_space.n,
                usable_ace_space.n,
                environment.action_space.n,
            )
        )

    def act(self, observation: BlackjackObservation) -> int:
        return int(self.get_q_values_for_observation(observation).argmax())

    def get_q_values_for_observation(
        self, observation: BlackjackObservation
    ) -> np.ndarray:
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


def run_iteration(agent: BlackjackAgent, environment: gym.Env) -> float:
    ret = 0
    observation = environment.reset()
    done = False

    while not done:
        action = agent.act(observation)
        observation_next, reward, done, _ = environment.step(action)
        ret += reward
        agent.update(observation, action, reward, observation_next)
        observation = observation_next

    return ret


def run_experiment(
    n_iterations: int, agent: BlackjackAgent, period: int = 10_000
) -> None:
    environment = gym.make("Blackjack-v0")
    rets = []

    for _ in range(n_iterations):
        ret = run_iteration(agent, environment)
        rets.append(ret)

    print(
        f"Average return per game (single {agent.name} agent): "
        + f"{sum(rets[-period:]) / period} from {period} games"
    )


def run() -> None:
    n_iterations = 100_000
    alpha = 0.01
    gamma = 0.01

    environment = gym.make("Blackjack-v0")
    null_agent = BlackjackAgent()
    random_agent = RandomAgent(environment)
    stick17_agent = Stick17Agent()
    q_learning_agent = QLearningAgent(environment, alpha, gamma)

    run_experiment(n_iterations, null_agent)
    run_experiment(n_iterations, random_agent)
    run_experiment(n_iterations, stick17_agent)
    run_experiment(n_iterations, q_learning_agent)


if __name__ == "__main__":
    run()
