from math import prod
import random

import gym
import torch as T
from torch import nn, optim
from torch.nn import functional as F

from config import GRID_ADDRESS, HIDDEN_WIDTH, INPUT_WIDTH, OUTPUT_WIDTH, NAME, VERSION
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


class QLearningAgent(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        hidden_width: int,
        alpha: float,
        gamma: float,
        min_epsilon: float,
        epsilon_reduction: float,
    ) -> None:
        super().__init__()
        self.name = NAME
        self.gamma = gamma  # discount rate
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction  # per action
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, output_width),
        )
        self.optimizer = optim.SGD(self.network.parameters(), lr=alpha)
        self.epsilon = nn.Linear(1, 1, bias=False)
        for p in self.epsilon.parameters():
            p.requires_grad = False

    def get_epsilon(self, train: bool) -> float:
        if train:
            epsilon = self.epsilon.weight.item()
            new_epsilon = max(self.min_epsilon, epsilon - self.epsilon_reduction)
            self.epsilon.weight.data = T.tensor([new_epsilon], requires_grad=False).data

        return self.epsilon.weight.item()

    def act(self, observation: CartPoleObservation, train: bool) -> int:
        if random.random() < self.get_epsilon(train):
            return random.randrange(2)

        processed_observation = T.tensor([observation], dtype=T.float32)
        return self.network(processed_observation).argmax().item()

    def update(
        self,
        observation: CartPoleObservation,
        action: int,
        reward: float,
        observation_next: CartPoleObservation,
    ) -> None:
        self.optimizer.zero_grad()
        processed_observation = T.tensor([observation], dtype=T.float32)
        processed_observation_next = T.tensor([observation_next], dtype=T.float32)

        current_q_value = self.network(processed_observation).squeeze()[action]
        next_q_value = self.network(processed_observation_next).squeeze().max()
        target_q_value = next_q_value * self.gamma + reward

        loss = F.mse_loss(current_q_value, target_q_value)
        loss.backward()
        self.optimizer.step()


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
        input_width=INPUT_WIDTH,
        output_width=OUTPUT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
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
