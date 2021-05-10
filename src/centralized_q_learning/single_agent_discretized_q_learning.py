import random

import gym  # type: ignore
import numpy as np

# (cart position, cart velocity, pole angle, pole angular velocity)
CartPoleObservation = tuple[float, float, float, float]
DiscretizedCartPoleObservation = tuple[int, int, int, int]


def clip(min_value, max_value, value):
    return max(min_value, min(max_value, value))


def discretize(observation: CartPoleObservation) -> DiscretizedCartPoleObservation:
    (
        _raw_cart_position,
        _raw_cart_velocity,
        raw_pole_angle,
        raw_pole_angular_velocity,
    ) = observation
    cart_position = 0  # not very useful
    cart_velocity = 0  # not very useful
    pole_angle = int(clip(0.0, 0.417, raw_pole_angle + 0.209) // (0.418 / 12))
    pole_angular_velocity = int(
        clip(0.0, 5.999, raw_pole_angular_velocity + 3.0) // (6.0 / 6)
    )
    return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)


class CartPoleAgent:
    def __init__(self) -> None:
        self.name = "null"

    def act(self, _observation: CartPoleObservation) -> int:
        return 0

    def update(
        self,
        _observation: CartPoleObservation,
        _action: int,
        _reward: float,
        _observation_next: CartPoleObservation,
    ) -> None:
        pass


class RandomAgent(CartPoleAgent):
    def __init__(self, environment: gym.Env) -> None:
        super().__init__()
        self.name = "random"
        self.action_space_size = environment.action_space.n

    def act(self, _observation: CartPoleObservation) -> int:
        return random.randrange(self.action_space_size)


class QLearningAgent(CartPoleAgent):
    def __init__(
        self,
        environment: gym.Env,
        alpha: float,
        gamma: float,
        epsilon: float,
        min_epsilon: float,
        epsilon_reduction: float,
    ) -> None:
        super().__init__()
        self.name = "q-learning"
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # initial exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction  # per action
        self.q_table = np.zeros((1, 1, 12, 6, environment.action_space.n))

    def act(self, observation: CartPoleObservation) -> int:
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_reduction)
        if random.random() < self.epsilon:
            return random.randrange(2)

        return int(self.get_q_values_for_observation(observation).argmax())

    def get_q_values_for_observation(
        self, observation: CartPoleObservation
    ) -> np.ndarray:
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = discretize(
            observation
        )
        return self.q_table[cart_position][cart_velocity][pole_angle][
            pole_angular_velocity
        ]

    def update(
        self,
        observation: CartPoleObservation,
        action: int,
        reward: float,
        observation_next: CartPoleObservation,
    ) -> None:
        q_values = self.get_q_values_for_observation(observation)
        max_next_q_value = self.get_q_values_for_observation(observation_next).max()
        q_values[action] = q_values[action] + self.alpha * (
            reward + self.gamma * max_next_q_value - q_values[action]
        )


def run_iteration(agent: CartPoleAgent, environment: gym.Env) -> float:
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


def run_experiment(n_iterations: int, agent: CartPoleAgent, period: int = 100) -> None:
    environment = gym.make("CartPole-v1")
    rets = []

    for i in range(n_iterations):
        ret = run_iteration(agent, environment)
        rets.append(ret)
        if sum(rets[-period:]) / period > 195.0:
            print(f"[single {agent.name} agent] Solved after {i+1} games")
            break
        # if (i + 1) % period == 0:
        #     print(
        #         f"[single {agent.name} agent] Average return per game: "
        #         + f"{sum(rets[-period:]) / period} from {period} games"
        #     )

    print(
        f"[single {agent.name} agent] Average return per game: "
        + f"{sum(rets[-period:]) / period} from {period} games"
    )


def run() -> None:
    n_iterations = 2000
    alpha = 0.1
    gamma = 1.0
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_reduction = 0.001

    environment = gym.make("CartPole-v1")
    null_agent = CartPoleAgent()
    random_agent = RandomAgent(environment)
    q_learning_agent = QLearningAgent(
        environment, alpha, gamma, epsilon, min_epsilon, epsilon_reduction
    )

    run_experiment(n_iterations, null_agent)
    run_experiment(n_iterations, random_agent)
    run_experiment(n_iterations, q_learning_agent)


if __name__ == "__main__":
    run()
