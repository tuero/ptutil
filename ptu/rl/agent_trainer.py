"""
File: agent_trainer.py
Author: Jake Tuero (tuero@ualberta.ca)
Date: July 15, 2021

Trainer object which handles the main training logic,
along with validation/testing
Calls appropriate callbacks
"""
from __future__ import annotations
from ptu.base_trainer import BaseTrainer

import os
import sys
import time
import torch
import gin
import gin.torch

from ptu.rl.base_environment import BaseEnvironment
from ptu.util.types import LoggingItem
from ptu.util.render import Render

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptu.rl.agent import PTUtilAgent
    from ptu.callbacks.callback_base import Callback
    from typing import List


@gin.configurable
class AgentTrainer(BaseTrainer):
    def __init__(
        self,
        env: BaseEnvironment,
        device: torch.device,
        num_episodes: int = 100,
        checkpoint_dir: str = "",
        cbs: List[Callback] = [],
        use_amp: bool = False,
    ):
        """Handles the training of RL agents on a given environment.
        For agent testing, there's an option for rendering of the environment for visualization.
        Passed arguments can be set by a gin config.
        NOTE: Use AMP at discretion, as it is not fully tested and there may be issues when
        combining with gradient clipping.

        Args:
            env: Environment to train on
            device: device for which all objects will reference
            num_episodes: Maximum number of episodes to train for
            checkpoint_dir: relative path (from script run) for checkpoint saving
            cbs: callbacks for the trainer to use
            use_amp: flag for whether to use automatic mixed precision
        """
        assert isinstance(env, BaseEnvironment), "Environment should extend (wrap) with BaseEnvironment"
        super().__init__(device, checkpoint_dir, cbs, use_amp)

        self.episode_end = num_episodes
        self.env = env
        self.step_episode = 0

    def _begin_fit(self, **kwargs) -> None:
        """Called before the fit process."""
        self.model.to(self.device)
        self.early_stoppage = False
        self.episode_loss = 0
        self.episode_reward = 0
        self.model.set_train()
        self.model.begin_fit(**kwargs)
        return self._run_callbacks("begin_fit")

    def _after_fit(self) -> None:
        """Called after the fit process (model is fully trained)."""
        self.model.after_fit()
        return self._run_callbacks("after_fit")

    def _begin_episode(self) -> None:
        """Called before each episode.
        This resets any counters/metrics for the episode.
        """
        self.step_episode = 0
        self.episode_loss = 0.0
        self.episode_reward = 0.0
        self.env.reset()
        self.model.begin_episode()
        return self._run_callbacks("begin_episode")

    def _begin_test_episode(self, **kwargs) -> None:
        """Called before the beginning of a episode during testing.
        Generally this is only called once.
        There may be special initialization needed for testing.
        """
        self.model.to(self.device)
        self.episode_reward = 0.0
        self.model.set_eval()
        self.env.reset()
        self.model.begin_test_episode(**kwargs)

    def _after_episode(self) -> None:
        """Called after each episode during training.
        This will store loss/reward information.
        """
        self.model.after_episode()
        self.episode_losses.append(self.episode_loss)
        self.episode_rewards.append(self.episode_reward)
        return self._run_callbacks("after_episode")

    def _after_test_episode(self) -> None:
        """Called after the episode during testing.
        Generally this is only called once.
        """
        self.model.after_test_episode()

    # After the backward pass is done on the loss
    def _after_backward(self) -> None:
        """Called after the backward pass is done on the loss.
        This is generally called after model.optimization_step().
        Gradient clipping will be called if the callback is set.
        """
        self.model.after_backward()
        return self._run_callbacks("after_backward")

    def _run_episode_step(self) -> None:
        """Runs a single step of the episode when training."""
        return self.model.run_episode_step()

    def _test_episode_step(self) -> None:
        """Runs a single step of the episode when testing."""
        return self.model.test_episode_step()

    # -----------------------------------
    #      Trainer specific methods
    # -----------------------------------

    def train_agent(self, agent: PTUtilAgent, **kwargs) -> None:
        """Handles all the necessary subroutines for training a PTUtilAgent.

        Args:
            agent: The agent model to train.
        """
        self.model = agent
        self.model.to(self.device)
        self.model.set_trainer(self)

        # Checkpoint not loaded, start from 0 and dump config params
        if not self.checkpoint_loaded:
            self.episode_start = 1
            self.step_global = 0
            self.logging_buffer.append(LoggingItem("INFO", gin.operative_config_str()))
            self.episode_losses = []
            self.episode_rewards = []
        self.checkpoint_loaded = False

        self._begin_fit(**kwargs)
        for ep in range(self.episode_start, self.episode_end + 1):
            self.step_ep = ep
            self._begin_episode()
            while not self.env.is_done():
                self.step_global += 1
                self.step_episode += 1
                with torch.cuda.amp.autocast(self.use_amp):
                    loss, reward = self._run_episode_step()
                self.episode_loss += loss
                self.episode_reward += reward
            self._after_episode()
            if self.early_stoppage:
                break
        self._after_fit()

    def test_agent(self, agent: PTUtilAgent, render: bool = False, **kwargs) -> None:
        """Tests the trained agent, along with an option to render the environment
        to visualize the agent.

        Args:
            agent: The trained PTUtilAgent agent
            render: Flag to visualize the agent
        """
        self.model = agent
        self.model.to(self.device)
        self.model.set_trainer(self)

        def _draw():
            if render:
                r.draw(self.env.state_to_image())
                time.sleep(0.5)

        if render:
            r = Render(600, 600)
        self._begin_test_episode(**kwargs)
        _draw()
        while not self.env.is_done():
            with torch.cuda.amp.autocast(self.use_amp):
                reward = self._test_episode_step()
                _draw()
            self.episode_reward += reward
        self._after_test_episode()

    def save_checkpoint(self, str_model: str = None) -> None:
        """Saves the trainer state along with the current agent model.
        This is automatically called when using the checkpoint callback.
        """
        model_dict = self.model.get_checkpoint()
        trainer_dict = {
            "ep": self.step_ep + 1,
            "step_global": self.step_global,
            "episode_losses": self.episode_losses,
            "episode_rewards": self.episode_rewards,
        }
        # check if directory exists, create if not
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        str_model = str_model if str_model is not None else str(self.model)
        checkpoint_file = os.path.join(self.checkpoint_dir, str_model + ".pt")
        torch.save({**model_dict, **trainer_dict}, checkpoint_file)
        self.save_checkpoint_flag = False

    def load_from_checkpoint(self, agent: PTUtilAgent) -> None:
        """Loads the trainer and model from checkpoint.
        This should be called externally from the main module.

        Args:
            model: Reference to an untrained model to load into
        """
        # Load model/checkpoint to device
        agent.to(self.device)
        checkpoint_file = os.path.join(self.checkpoint_dir, str(agent) + ".pt")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            # Set internal variables from checkpoint
            self.episode_start = checkpoint["ep"]
            self.step_global = checkpoint["step_global"]
            self.episode_losses = checkpoint["episode_losses"]
            self.episode_rewards = checkpoint["episode_rewards"]
            agent.load_from_checkpoint(checkpoint)
            # Set flags and cleanup
            self.checkpoint_loaded = True
            self.logging_buffer.append(LoggingItem("INFO", "Loading from checkpoint."))
            del checkpoint
        except Exception as e:
            print(str(e))
            sys.exit("Error: Unable to open config file {}".format(checkpoint_file))
