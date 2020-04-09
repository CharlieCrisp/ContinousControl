import numpy as np

from ppo_continuous_control.ppo_algorithm import calculate_future_rewards


def test_calculate_future_rewards():
    # Arrange
    rewards = np.array([0, 1, 0, 1, 1])
    expected_future_rewards = np.array([3, 3, 2, 2, 1])

    # Act
    actual_future_rewards = calculate_future_rewards(rewards)

    # Assert
    assert all(actual_future_rewards == expected_future_rewards)
