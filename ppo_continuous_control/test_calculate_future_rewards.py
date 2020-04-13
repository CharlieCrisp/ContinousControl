import torch
import numpy as np

from ppo_continuous_control.ppo_algorithm import calculate_future_rewards


def test_calculate_future_rewards_when_gamma_is_one():
    # Arrange
    rewards = np.array([0, 1, 0, 1, 1])
    dones = np.array([False, False, False, False, True])
    expected_future_rewards = torch.tensor([3, 3, 2, 2, 1], dtype=torch.float64)

    # Act
    actual_future_rewards = calculate_future_rewards(rewards, dones, gamma=1)

    # Assert
    assert torch.allclose(
        actual_future_rewards, expected_future_rewards
    ), f"Expected {expected_future_rewards} but got {actual_future_rewards}"


def test_calculate_future_rewards_when_gamma_is_less_than_one():
    # Arrange
    rewards = np.array([0, 1, 0, 1, 1])
    dones = np.array([False, False, False, False, True])

    gamma = 0.99
    expected_discounted_rewards = torch.tensor(
        [
            1 * gamma + 1 * gamma ** 3 + 1 * gamma ** 4,
            1 + 1 * gamma ** 2 + 1 * gamma ** 3,
            1 * gamma + 1 * gamma ** 2,
            1 + 1 * gamma,
            1,
        ],
        dtype=torch.float64,
    )

    # Act
    actual_future_rewards = calculate_future_rewards(rewards, dones, gamma=gamma)

    # Assert
    assert torch.allclose(
        actual_future_rewards, expected_discounted_rewards
    ), f"Expected {expected_discounted_rewards} but got {actual_future_rewards}"


def test_calculate_future_rewards_when_there_are_multiple_trajectories():
    # Arrange
    rewards = np.array([0, 1, 0, 1, 1, 1])
    dones = np.array([False, False, True, False, False, True])

    gamma = 0.99
    expected_discounted_rewards = torch.tensor(
        [1 * gamma, 1, 0, 1 + 1 * gamma + 1 * gamma ** 2, 1 + 1 * gamma, 1,],
        dtype=torch.float64,
    )

    # Act
    actual_future_rewards = calculate_future_rewards(rewards, dones, gamma=gamma)

    # Assert
    assert torch.allclose(
        actual_future_rewards, expected_discounted_rewards
    ), f"Expected {expected_discounted_rewards} but got {actual_future_rewards}"
