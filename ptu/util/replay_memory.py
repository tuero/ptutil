"""
File: replay_memory.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: Oct 10, 2020

Replay buffer to encourage indpendence between samples.
Includes both a standard uniform replay memory and a prioritized replay memory.
Source code for the prioritized version is taken from https://github.com/rlcode/per
"""

from collections import namedtuple
import random
from typing import NamedTuple, Tuple
import numpy as np

from ptu.util.sum_tree import SumTree


# Default container item in the memory
Transition = namedtuple("Transion", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int, min_sample_size: int, TransitionSampleType: NamedTuple = Transition):
        """Standard uniform replay memory.

        Args:
            capacity: Maximum storage size
            min_sample_size: Minimum number of samples stored before sampling is allowed
            TransitionSampleType: Namedtuple structure of items in the container
        """
        self._capacity = capacity
        self._min_sample_size = min_sample_size
        self.TransitionSampleType = TransitionSampleType
        self.clear()

    def clear(self) -> None:
        """Reset the replay memory by removing all items."""
        self._memory = []
        self._position = 0

    def can_sample(self) -> bool:
        """Check for whether there is enough stored items to sample.

        Returns:
            Returns true if the replay memory can be sampled.
        """
        return len(self._memory) > self._min_sample_size

    def push(self, *args) -> None:
        """Store the passed arugments into the memory.
        This assumes the passed arugments match the TransitionSampleType as set in the initializer.

        Args:
            args: Contents to store
        """
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = self.TransitionSampleType(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int) -> NamedTuple:
        """Returns a batched sample from the replay memory.

        Args:
            batch_size: Number of items to sample

        Returns:
            A named tuple, where each item of the named tuple contains a batch of that subitem.
        """
        transitions = random.sample(self._memory, batch_size)
        return self.TransitionSampleType(*zip(*transitions))

    def __len__(self):
        return len(self._memory)


class PrioritizedReplayMemory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity: int, min_sample_size: int, TransitionSampleType: NamedTuple = Transition):
        """Prioritized replay memory.

        Args:
            capacity: Maximum storage size
            min_sample_size: Minimum number of samples stored before sampling is allowed
            TransitionSampleType: Namedtuple structure of items in the container
        """
        self.capacity = capacity
        self.min_sample_size = min_sample_size
        self.TransitionSampleType = TransitionSampleType
        self.clear()

    def clear(self) -> None:
        """Reset the replay memory by removing all items."""
        self.tree = SumTree(self.capacity)

    def can_sample(self) -> bool:
        """Check for whether there is enough stored items to sample.

        Returns:
            Returns true if the replay memory can be sampled.
        """
        return self.tree.n_entries > self.min_sample_size

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, error, *args):
        """Store the passed arugments into the memory.
        This assumes the passed arugments match the TransitionSampleType as set in the initializer.

        Args:
            args: Contents to store
        """
        # Capacity and rolling handeled by the tree
        p = self._get_priority(error)
        self.tree.add(p, self.TransitionSampleType(*args))

    def sample(self, batch_size: int) -> Tuple[NamedTuple, np.ndarray, np.ndarray]:
        """Returns a batched sample from the replay memory.

        Args:
            batch_size: Number of items to sample

        Returns:
            A named tuple, where each item of the named tuple contains a batch of that subitem.
            Numpy array of indices of the batch
            Numpy array of the sample weights
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        batch = Transition(*zip(*batch))
        return batch, idxs, is_weight

    def update(self, idx: int, error: float) -> None:
        """Update the weights of the sample given the prediction error.

        Args:
            idx: Index to update
            error: Prediction error, which will be used to update the tree
        """
        p = self._get_priority(error)
        self.tree.update(idx, p)
