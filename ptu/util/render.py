"""
File: render.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: July 18, 2020

Render functions to aide in reinforcement learning visualizations.
"""

import cv2
import numpy as np
import pygame


class Render:
    def __init__(self, screen_width: int, screen_height: int):
        """Render object for drawing images from the given environment.

        Args:
            screen_width: The screen width in pixels
            screen_height: The screen height in pixels
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))

    def check_for_exit(self) -> bool:
        """Checks if exit has been called."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

    def draw(self, observation: np.ndarray):
        """Draws a given numpy array. Assumes the given array is in (CHW) format.

        Args:
            observation: Numpy array observation representing the picture to draw.
        """
        # Reshape observation
        observation = cv2.resize(
            observation, dsize=(self._screen_width, self._screen_width), interpolation=cv2.INTER_NEAREST
        ).astype("uint8")
        observation = np.transpose(observation, (1, 0, 2))
        observation_surface = pygame.surfarray.make_surface(observation)
        self._screen.blit(observation_surface, (0, 0))
        pygame.display.update()

    def close(self):
        """Closes the pygame window."""
        pygame.quit()
