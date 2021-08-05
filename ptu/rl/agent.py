"""
File: agent.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: July 15, 2021

Module object which contains the agent model,
interfaces with agent_trainer and callbacks
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import ray
from ptu.callbacks.grad_clipping import GradClipping

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Callable, Dict, Any, Tuple
    from ptu.rl.agent_trainer import AgentTrainer


def maybe_parallelize(function: Callable, arg_list: Iterable) -> Iterable:
    """Optional way to parallelize processing of a batch.
    This uses ray if initialized, or just calls linearly over the batch.
    NOTE: Using something like ray may be slower if the overhead is greater than the
    actual processing work.

    Args:
        function: The processing function to call
        arg_list: The batch of items to pass to the given function

    Returns:
        A list of the returned results of the function being applied to arg_list
    """
    if ray.is_initialized():
        ray_fun = ray.remote(function)
        return ray.get([ray_fun.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]


class PTUtilAgent(nn.Module):
    def __init__(self, *args, **kwargs):
        """Agent object, which holds the necessary models for the agent and is called by the trainer.
        This extends nn.Module, so that the entire sub-models can be treated as one large one.
        NOTE: Initializer should initialize the optimizers inside self.optimizer_infos,
        so that the optimizer states can be saved/loaded for checkpointing.
        """
        super().__init__(*args, **kwargs)
        self.optimizer_infos = None

    def set_trainer(self, trainer: AgentTrainer):
        """Saves a trainer reference so we can pull trainer info"""
        self.trainer = trainer
        self.device = self.trainer.device

    def set_device(self, device: torch.device):
        self.device = device

    def __str__(self):
        """String name for checkpoints."""
        raise NotImplementedError

    def get_checkpoint(self) -> Dict[str, Any]:
        """Creats a checkpoint dictionary for the trainer object.
        If this method is overwritten, the optimizers in self.optimizer_info needs to be manually stored.
        This is called during trainer.save_checkpoint().

        Returns:
            Dictionary holding stored state info
        """
        model_dict = {"model": self.state_dict()}
        for i, optimizer_info in enumerate(self.optimizer_infos):
            model_dict["optimizer_{}".format(i)] = optimizer_info.optimizer.state_dict()
            model_dict["scheduler_{}".format(i)] = optimizer_info.scheduler.state_dict()
        return model_dict

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Sets the model state from the given checkpoint.
        If self.get_checkpoint() is overwritten, this must also be overwritten to match.

        Args:
            checkpoint: Checkpoint given from the trainer object.
        """
        self.load_state_dict(checkpoint["model"])
        for i, optimizer_info in enumerate(self.optimizer_infos):
            optimizer_info.optimizer.load_state_dict(checkpoint["optimizer_{}".format(i)])
            optimizer_info.scheduler.load_state_dict(checkpoint["optimizer_{}".format(i)])

    def set_train(self) -> None:
        """Puts the model into train mode."""
        self.train(True)

    def set_eval(self) -> None:
        """Puts the model into eval mode."""
        self.train(False)

    def process_batch(
        self, batch: NamedTuple, state_tensor_fnc=lambda x: x
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Process a batch pulled from the replay memory.
        This assumes items are stored as a namedtuple and closely follows replay_memory.Transition
        (i.e. contains state, action, next_state, and reward subitems).

        Args:
            batch: Namedtuple in which each sub-item contains all of those sub-items for the batch
            state_tensor_fnc: Function to process each state in memory before it gets passed into the network (optional)

        Returns:
            state batch
            action batch
            reward batch
            non-final state mask for each item in the batch
            non-final next states for each item in the batch which is not final
        """
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )

        if True not in non_final_mask:
            non_final_next_states = None
        else:
            non_final_next_states = torch.cat(
                maybe_parallelize(state_tensor_fnc, [s for s in batch.next_state if s is not None])
            ).to(self.device)
        state_batch = torch.cat(maybe_parallelize(state_tensor_fnc, [s for s in batch.state if s is not None])).to(
            self.device
        )

        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states

    def calc_target_values(
        self, policy_net: nn.Module, state_batch: torch.tensor, action_batch: torch.tensor
    ) -> torch.tensor:
        """Finds the target values.

        Args:
            policy_net: The network whose outputs are the state-action values
            state_batch: The batch of states to evaulate
            action_batch: The actions being applied for each state in the batch

        Returns:
            Action values for each item in the batch
        """
        return policy_net(state_batch).gather(1, action_batch)

    def calc_expected_target_values(
        self,
        target_net: nn.Module,
        non_final_next_states: torch.tensor,
        non_final_mask: torch.tensor,
        reward_batch: torch.tensor,
        gamma: float,
    ) -> torch.tensor:
        """Find the expected target values from a target network.

        Args:
            target_net: The reference target network whsoe outputs are the expected state-action values
            non_final_next_states: Batch of states which are not final
            non_funal_mask: Mask of batch whose next states are not final
            reward_batch: Batch of reward observed
            gamma: The discounting factor.

        Returns:
            Expected action values of the next state for each item in the batch
        """
        with torch.cuda.amp.autocast(self.trainer.use_amp):
            next_state_values = torch.zeros(len(non_final_mask), device=self.device)
            if non_final_next_states is not None:
                # Hacky version to do whats commented out below.
                # For some reason, pytorch amp can't deduce the type, so we need a manual cast
                # May change this if I can find a better way.
                net_out = target_net(non_final_next_states).max(1)[0].detach()
                next_state_values = next_state_values.to(net_out.dtype)
                next_state_values[non_final_mask] = net_out
                # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * gamma) + reward_batch
            return expected_state_action_values.unsqueeze(1)

    def epsilon_greedy_selection(
        self, state: np.ndarray, num_actions: int, network: nn.Module, epsilon: float
    ) -> torch.tensor:
        """Performs an epsilon-greedy action selection.

        Args:
            state: The current state
            num_actions: The number of possible actions (used in random selection)
            network: The network to query actions from
            epsilon: The level of greediness

        Returns:
            The selected action
        """
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                return network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(num_actions)]], device=self.device, dtype=torch.long)

    def optimization_step(
        self, network: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.tensor
    ) -> float:
        """Perform an optimization step of the given network with respect to the given loss.
        This should be manually called in self.training_step().
        If gradient clipping callback is set on the trainer, then it will get called here on the given model.

        Args:
            network: The network to update (needed for any callbacks which need the networks parameters)
            optimizer: The optimizer whose params should match the given network's params
            loss: Loss for updating the network

        Returns:
            The loss value.
        """
        optimizer.zero_grad()
        self.trainer.loss = loss
        _loss = loss.item()
        self.trainer.grad_clip_buffer.append(network)
        if self.trainer.use_amp:
            self.trainer.scaler.scale(loss).backward()
            if True in [isinstance(c, GradClipping) for c in self.trainer.cbs]:
                self.trainer.scaler.unscale_(optimizer)
            self.trainer.scaler.unscale_(optimizer)
            self.trainer._after_backward()  # Will run grad_clipping if necessary
            self.trainer.scaler.step(optimizer)
            self.trainer.scaler.update()
        else:
            loss.backward()
            self.trainer._after_backward()  # Will run grad_clipping if necessary
            optimizer.step()
        return _loss

    def soft_update_network(self, local_network: nn.Module, target_network: nn.Module, tau: float) -> None:
        """Perform a soft update of bringing the target network towards the local policy network.

        Args:
            local_network: The local policy network
            target_network: The target network to update
            tau: The amount of change for the soft update
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update_network(self, local_network: nn.Module, target_network: nn.Module) -> None:
        """Perform a soft update of setting the target network to the local policy network.

        Args:
            local_network: The local policy network
            target_network: The target network to update
        """
        target_network.load_state_dict(local_network.state_dict())

    # -----------------------------------
    # Callback matching steps for trainer
    # -----------------------------------

    def run_episode_step(self) -> Tuple[float, float]:
        """Called at each step during the episode.
        The user implementation should handle interacting with the environment, storing samples,
        and running an optimizaiton pass

        Returns:
            The loss for the optimization step run during the episode step
            The reward from the environment for the episode step
        """
        raise NotImplementedError

    def test_episode_step(self) -> None:
        """Called at each step during the test episode.
        There is no requirements for returned values.
        """
        raise NotImplementedError

    def begin_fit(self, **kwargs) -> None:
        """Called before the fit process."""
        return

    def after_backward(self) -> None:
        """Called after the backward pass is done on the loss."""
        return

    def after_fit(self) -> None:
        """Called after the agent model is trained."""
        return

    def begin_episode(self) -> None:
        """Called at the beginning of each episode during training."""
        return

    def begin_test_episode(self, **kwargs) -> None:
        """Called at the beginning of each episode during testing."""
        return

    def after_episode(self) -> None:
        """Called at the end of each episode during training."""
        return

    def after_test_episode(self) -> None:
        """Called at the end of each episode during testing."""
        return
