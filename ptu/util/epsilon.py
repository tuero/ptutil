"""
File: epsilon.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: July 20, 2021

Helper methods for epsilon rate/decay for reinforcement learning
"""

import numpy as np
from typing import Callable, Tuple


def linear_decay_rate(start_rate: float, end_rate: float, n_expected_updates: int) -> float:
    """Find the decay rate following a linear schedule.

    Args:
        start_rate: Rate to start at (highest)
        end_rate: Rate to end at (lowest)
        n_expected_updates: Length to decay over

    Returns:
        The decay rate to apply at every epoch
    """
    return (start_rate - end_rate) / n_expected_updates


def exponential_decay_rate(start_rate: float, end_rate: float, n_expected_updates: int) -> float:
    """Find the decay rate following a exponential schedule.

    Args:
        start_rate: Rate to start at (highest)
        end_rate: Rate to end at (lowest)
        n_expected_updates: Length to decay over

    Returns:
        The decay rate to apply at every epoch
    """
    return np.exp(np.log(end_rate / start_rate) / n_expected_updates)


def decay_linear(current_rate: float, decay: float, min_rate: float) -> float:
    """Calculates the new epsilon value following a linear decay.

    Args:
        current_rate: The current rate
        decay: The decay rate
        min_rate: The minimum value the rate can be

    Returns:
        The new rate value
    """
    return max(current_rate - decay, min_rate)


def decay_exponential(current_rate: float, decay: float, min_rate: float) -> float:
    """Calculates the new epsilon value following an exponential decay.

    Args:
        current_rate: The current rate
        decay: The decay rate
        min_rate: The minimum value the rate can be

    Returns:
        The new rate value
    """
    return max(current_rate * decay, min_rate)


def get_epsilon_update(schedule: str, start: float, end: float, length: int) -> Tuple[float, Callable]:
    """Gets the decay rate and decay function corresponding to the given schedule type.

    Args:
        schedule: Schedule type (linear, exponential)
        start: Starting rate value
        end: Ending rate value
        length: Length to decay over

    Returns
        The decay value
        The decay update function
    """
    if schedule == "linear":
        return linear_decay_rate(start, end, length), decay_linear
    elif schedule == "exponential":
        return exponential_decay_rate(start, end, length), decay_exponential
    else:
        raise ValueError("Unknown epsilon mode.")
