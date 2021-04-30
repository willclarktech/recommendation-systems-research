import gym  # type: ignore
import numpy as np

# (current sum, dealer card, usable ace)
BlackjackObservation = tuple[int, int, bool]


class QLearningAgent:
    def __init__(self, environment: gym.Env, alpha: float, gamma: float) -> None:
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


AgentEnvironments = list[tuple[QLearningAgent, gym.Env]]


def calculate_pooled_diff(
    q_table_checkpoint: np.ndarray, agents: list[QLearningAgent]
) -> np.ndarray:
    if len(agents) == 0:
        return np.zeros_like(q_table_checkpoint)

    agent_diffs = np.array([agent.q_table - q_table_checkpoint for agent in agents])
    return agent_diffs.mean(axis=0)  # type: ignore


def run_iteration(agent: QLearningAgent, environment: gym.Env) -> float:
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


def run_epoch(
    n_iterations: int,
    agent_environments: AgentEnvironments,
    q_table_checkpoint: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    rets = []

    for agent, environment in agent_environments:
        agent.q_table = np.array(q_table_checkpoint)
        for _ in range(n_iterations):
            ret = run_iteration(agent, environment)
            rets.append(ret)

    pooled_diff = calculate_pooled_diff(
        q_table_checkpoint, [agent for agent, _ in agent_environments]
    )
    return pooled_diff, rets


def run_experiment(
    n_agents: int,
    n_epochs: int,
    n_iterations_per_epoch: int,
    alpha: float,
    gamma: float,
    period: int = 10_000,
) -> None:
    rets = []
    agent_environments: AgentEnvironments = []
    for _ in range(n_agents):
        environment = gym.make("Blackjack-v0")
        agent = QLearningAgent(environment, alpha=alpha, gamma=gamma)
        agent_environments.append((agent, environment))

    q_table_checkpoint = np.array(agent_environments[0][0].q_table)

    for _ in range(n_epochs):
        diff, epoch_rets = run_epoch(
            n_iterations_per_epoch, agent_environments, q_table_checkpoint
        )
        q_table_checkpoint += diff
        rets += epoch_rets

    print(
        f"Average return per game ({n_agents} {agent.name} agents): "
        + f"{sum(rets[-period:]) / period} from {period} games"
    )


def run() -> None:
    n_epochs = 3000
    n_iterations_per_epoch = 10
    n_agents = 3
    alpha = 0.005
    gamma = 0.01

    run_experiment(n_agents, n_epochs, n_iterations_per_epoch, alpha, gamma)


if __name__ == "__main__":
    run()
