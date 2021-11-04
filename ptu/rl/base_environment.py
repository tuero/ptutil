"""
File: base_environment.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: October 18, 2020

Base environment handler which each environment will inherit from
"""

from typing import Tuple, Any
import numpy as np
import torch


class BaseEnvironment:
    def num_actions(self) -> int:
        """Get the number of discrete actions for the environment

        Returns:
            Number of actions
        """
        raise NotImplementedError

    def obs_shape(self) -> Tuple[int]:
        """Observation shape, used for network configurations

        Returns:
            Array shape of observations, (c,h,w) for image observations
        """
        raise NotImplementedError

    def state_to_tensor(self, state: np.ndarray) -> torch.tensor:
        """Process the state observation into a state tensor representation

        Inputs:
            state: state observation from environment

        Returns:
            modified passed tensor for network
        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Reset the environment

        Returns:
            The initial state
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns true if the environment episode is over, false otherwise

        Returns:
            True if episode done, false otherwise
        """
        raise NotImplementedError

    def get_current_state(self) -> np.ndarray:
        """Get the current state of the environment

        Returns:
            The current state
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Set the environment forward after performing the given action

        Inputs:
            action: action index to perform

        Returns:
            observation of next state
            Reward earned during the step
            Flag for environment end state
        """
        raise NotImplementedError

    def state_to_image(self, state: Any = None) -> np.ndarray:
        """Convert the current state to an image used for neural network models

        Args:
            state: The state to process (optional). If not given, uses current state.

        Returns:
            image representing current state (c, h, w)

        """
        raise NotImplementedError

    def render(self, state: Any = None) -> np.ndarray:
        """Convert the current state to an image for render drawing to screen

        Args:
            state: The state to process (optional). If not given, uses current state.

        Returns:
            image representing current state (h, w, c)

        """
        raise NotImplementedError
