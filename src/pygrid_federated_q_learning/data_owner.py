import gym  # type: ignore
import torch as T
from torch import nn

from config import BLACKJACK_DIMS, GRID_ADDRESS, NAME, VERSION
from utils import (
    calculate_diff,
    get_model_params,
    retrieve_model_params,
    send_auth_request,
    send_cycle_request,
    send_diff_report,
    set_params,
)


# (current sum, dealer card, usable ace)
BlackjackObservation = tuple[int, int, bool]


class QLearningAgent(nn.Module):
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.network = nn.Linear(32 * 11 * 2 * 2, 1, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        nn.init.zeros_(self.network.weight)

    def act(self, observation: BlackjackObservation) -> int:
        output = self.get_q_values_for_observation(observation)
        return int(output.argmax())

    def get_q_values_for_observation(
        self, observation: BlackjackObservation
    ) -> T.Tensor:
        q_table = self.network.weight.reshape(BLACKJACK_DIMS)
        current_sum, dealer_card, usable_ace = observation
        return q_table[current_sum][dealer_card][int(usable_ace)]

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


def run_epoch(n_iterations: int, agent: QLearningAgent, train=True):
    environment = gym.make("Blackjack-v0")
    rets = []

    for _ in range(n_iterations):
        ret = run_iteration(agent, environment, train)
        rets.append(ret)

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
    n_train_iterations = client_config["n_train_iterations"]
    n_test_iterations = client_config["n_test_iterations"]

    downloaded_params = get_model_params(GRID_ADDRESS, worker_id, request_key, model_id)

    local_agent = QLearningAgent(alpha=alpha, gamma=gamma)
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
