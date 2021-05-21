from math import prod
import random

import gym
import torch as T
from torch import nn

from config import CARTPOLE_DIMS, GRID_ADDRESS, NAME, VERSION
from utils import (
    calculate_diff,
    get_model_params,
    retrieve_model_params,
    send_auth_request,
    send_cycle_request,
    send_diff_report,
    set_params,
)

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


class QLearningAgent(nn.Module):
    def __init__(
        self, alpha: float, gamma: float, min_epsilon: float, epsilon_reduction: float
    ) -> None:
        super().__init__()
        self.name = "q-learning"
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction  # per action
        self.network = nn.Linear(prod(CARTPOLE_DIMS), 1, bias=True)
        for p in self.parameters():
            p.requires_grad = False

    def get_epsilon(self, train: bool) -> float:
        if train:
            epsilon = self.network.bias.item()
            new_epsilon = max(self.min_epsilon, epsilon - self.epsilon_reduction)
            self.network.bias.data = T.tensor([new_epsilon], requires_grad=False).data

        return self.network.bias.item()

    def get_q_values_for_observation(
        self, observation: CartPoleObservation
    ) -> T.Tensor:
        q_table = self.network.weight.reshape(CARTPOLE_DIMS)
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = discretize(
            observation
        )
        return q_table[cart_position][cart_velocity][pole_angle][pole_angular_velocity]

    def act(self, observation: CartPoleObservation, train: bool) -> int:
        if random.random() < self.get_epsilon(train):
            return random.randrange(2)

        return int(self.get_q_values_for_observation(observation).argmax())

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


def run_iteration(agent: QLearningAgent, environment: gym.Env, train: bool) -> float:
    ret = 0.0
    observation = environment.reset()
    done = False

    while not done:
        action = agent.act(observation, train)
        observation_next, reward, done, _ = environment.step(action)
        ret += reward
        if train:
            agent.update(observation, action, reward, observation_next)
        observation = observation_next

    return ret


def run_epoch(n_iterations: int, agent: QLearningAgent, train=True, period=100):
    environment = gym.make("CartPole-v1")
    rets = []

    for i in range(n_iterations):
        ret = run_iteration(agent, environment, train)
        rets.append(ret)
        if (i + 1) % period == 0:
            print(
                f"[federated {agent.name} agent] Epoch {i + 1} Average return per game: "
                + f"{sum(rets[-period:]) / period} from {period} games"
            )

    return list(agent.parameters()), rets


def main() -> None:
    auth_response = send_auth_request(GRID_ADDRESS, NAME, VERSION)
    worker_id = auth_response["data"]["worker_id"]

    cycle_response = send_cycle_request(GRID_ADDRESS, NAME, VERSION, worker_id)
    request_key = cycle_response["data"]["request_key"]
    model_id = cycle_response["data"]["model_id"]
    client_config = cycle_response["data"]["client_config"]
    alpha = client_config["alpha"]
    gamma = client_config["gamma"]
    min_epsilon = client_config["min_epsilon"]
    epsilon_reduction = client_config["epsilon_reduction"]
    n_train_iterations = client_config["n_train_iterations"]
    n_test_iterations = client_config["n_test_iterations"]

    downloaded_params = get_model_params(GRID_ADDRESS, worker_id, request_key, model_id)

    local_agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        min_epsilon=min_epsilon,
        epsilon_reduction=epsilon_reduction,
    )
    set_params(local_agent, downloaded_params)

    _, pre_rets = run_epoch(n_test_iterations, local_agent, train=False)
    print(f"Pre-training performance: {sum(pre_rets) / n_test_iterations}")

    trained_params, _ = run_epoch(n_train_iterations, local_agent, train=True)

    _, post_rets = run_epoch(n_test_iterations, local_agent, train=False)
    print(f"Post-training performance: {sum(post_rets) / n_test_iterations}")

    diff = calculate_diff(downloaded_params, trained_params)
    send_diff_report(GRID_ADDRESS, worker_id, request_key, diff)

    new_model_params = retrieve_model_params(GRID_ADDRESS, NAME, VERSION)
    set_params(local_agent, new_model_params)

    _, updated_rets = run_epoch(n_test_iterations, local_agent, train=False)
    print(f"Updated model performance: {sum(updated_rets) / n_test_iterations}")


if __name__ == "__main__":
    main()
