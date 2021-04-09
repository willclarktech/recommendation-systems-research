import gym  # type: ignore
import numpy as np
from numpy.testing import assert_array_equal

from multi_agent_q_learning import QLearningAgent, calculate_pooled_diff

DEFAULT_ALPHA = 0.01
DEFAULT_GAMMA = 0.01


def create_test_checkpoint() -> np.ndarray:
    return np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )


def create_test_environment() -> gym.Env:
    return gym.make("Blackjack-v0")


def test_calculate_pooled_diff_zero_agents() -> None:
    checkpoint = create_test_checkpoint()
    agents: list[QLearningAgent] = []
    diff = calculate_pooled_diff(checkpoint, agents)
    assert_array_equal(diff, np.zeros((2, 3, 4)))


def test_calculate_pooled_diff_one_agent() -> None:
    checkpoint = create_test_checkpoint()
    environment = create_test_environment()
    agent = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent.q_table = checkpoint + 1.0
    diff = calculate_pooled_diff(checkpoint, [agent])
    assert_array_equal(diff, np.ones_like(checkpoint))


def test_calculate_pooled_diff_two_agents() -> None:
    checkpoint = create_test_checkpoint()
    environment = create_test_environment()
    agent1 = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent2 = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent1.q_table = checkpoint + 1.0
    agent2.q_table = checkpoint + 3.0
    diff = calculate_pooled_diff(checkpoint, [agent1, agent2])
    assert_array_equal(diff, np.ones_like(checkpoint) * 2.0)


def test_calculate_pooled_diff_three_agents() -> None:
    checkpoint = create_test_checkpoint()
    environment = create_test_environment()
    agent1 = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent2 = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent3 = QLearningAgent(environment, DEFAULT_ALPHA, DEFAULT_GAMMA)
    agent1.q_table = checkpoint + 1.5
    agent2.q_table = checkpoint + 4.5
    agent3.q_table = checkpoint + 9.0
    diff = calculate_pooled_diff(checkpoint, [agent1, agent2, agent3])
    assert_array_equal(diff, np.ones_like(checkpoint) * 5.0)
