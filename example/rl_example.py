import argparse
import gin
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torchvision.transforms.functional import InterpolationMode

from ptu.util.replay_memory import ReplayMemory
from ptu.rl.base_environment import BaseEnvironment
from ptu.util.epsilon import get_epsilon_update
from ptu.util.types import LoggingItem
from ptu.util.types import OptimizerInfo
from ptu.util.types import MetricsItem
from ptu.util.types import MetricFramework, MetricType, Mode

from ptu.rl.agent_trainer import AgentTrainer
from ptu.rl.agent import PTUtilAgent
from ptu.callbacks.logger import Logger
from ptu.callbacks.checkpoint import Checkpoint
from ptu.callbacks.grad_clipping import GradClipping
from ptu.callbacks.tracker_tensorboard import TrackerTensorboard


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env, resize):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    return screen


class MountainCarEnv(BaseEnvironment):
    def __init__(self):
        self.env = gym.make("CartPole-v0").unwrapped
        self.resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=InterpolationMode.BICUBIC), T.ToTensor()])
        self.n_actions = self.env.action_space.n
        self.reset()
        self._obs_shape = get_screen(self.env, self.resize).shape

    def num_actions(self):
        return self.n_actions

    def obs_shape(self):
        return self._obs_shape

    def state_to_tensor(self, state):
        return torch.from_numpy(state).unsqueeze(0).float()

    def reset(self):
        self.env.reset()
        self.last_screen = get_screen(self.env, self.resize)
        self.current_screen = get_screen(self.env, self.resize)
        self.done = False

    def is_done(self):
        return self.done

    def get_current_state(self):
        return self.current_screen - self.last_screen

    def step(self, action):
        _, reward, self.done, _ = self.env.step(action)
        self.last_screen = self.current_screen
        self.current_screen = get_screen(self.env, self.resize)
        next_state = None if self.done else self.get_current_state()
        return next_state, reward, self.done


class DQNAgent(PTUtilAgent):
    def __init__(self, env: BaseEnvironment):
        super().__init__()
        self.num_actions = env.num_actions()
        self.batch_size = 64
        self.replay_memory = ReplayMemory(10000, 256)
        self.epsilon = 0.9
        self.gamma = 0.99
        self.epsilon_decay, self.epsilon_update = get_epsilon_update("exponential", self.epsilon, 0.05, 10000)

        _, screen_height, screen_width = env.obs_shape()
        self.policy_network = DQN(screen_height, screen_width, self.num_actions)
        self.target_network = DQN(screen_height, screen_width, self.num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.optimizer_infos = [OptimizerInfo(optimizer=self.optimizer)]

    def __str__(self):
        return "dqn-agent"

    def after_episode(self):
        # Hard update
        if self.trainer.step_ep % 15 == 0 and self.replay_memory.can_sample():
            self.hard_update_network(self.policy_network, self.target_network)

        # Save checkpoint flag
        if self.trainer.step_ep % 10 == 0 and self.replay_memory.can_sample():
            self.trainer.save_checkpoint_flag = True

        # Log metrics
        ep_loss = self.trainer.episode_loss / self.trainer.step_episode
        base_str = "ep: {:>4}, ep len: {:>4}, global_steps: {:>8}, reward: {:>8.1f}, loss: {:>10.6f}: epsilon: {:.2f}"
        logg_str = base_str.format(
            self.trainer.step_ep,
            self.trainer.step_episode,
            self.trainer.step_global,
            self.trainer.episode_reward,
            ep_loss,
            self.epsilon,
        )
        self.trainer.logging_buffer.append(LoggingItem("INFO", logg_str))
        self.trainer.metrics_buffer.append(
            MetricsItem(
                MetricFramework.tensorboard,
                MetricType.scalar,
                Mode.train,
                ("ep_reward", self.trainer.episode_reward, self.trainer.step_ep),
            )
        )

    def select_action(self, state, epsilon):
        with torch.no_grad():
            state_tensor = self.trainer.env.state_to_tensor(state).to(self.trainer.device)
            return self.epsilon_greedy_selection(state_tensor, self.num_actions, self.policy_network, epsilon)

    def loss_fnc(self, state_action_values, expected_state_action_values):
        return F.smooth_l1_loss(state_action_values, expected_state_action_values)

    def optimize(self):
        batch = self.replay_memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = self.process_batch(
            batch, self.trainer.env.state_to_tensor
        )
        # Current Q values from policy network
        state_action_values = self.calc_target_values(self.policy_network, state_batch, action_batch)
        # Expected Q values of next states max Q
        expected_state_action_values = self.calc_expected_target_values(
            self.target_network, non_final_next_states, non_final_mask, reward_batch, self.gamma
        )
        loss = self.loss_fnc(state_action_values, expected_state_action_values)
        return self.optimization_step(self.policy_network, self.optimizer_infos[0].optimizer, loss)

    def run_episode_step(self):
        state = self.trainer.env.get_current_state()
        action = self.select_action(state, self.epsilon).item()
        next_state, reward, done = self.trainer.env.step(action)

        self.replay_memory.push(state, torch.tensor([action]), next_state, torch.tensor(reward))

        loss = 0.0
        if self.replay_memory.can_sample():
            loss = self.optimize()
            self.epsilon = self.epsilon_update(self.epsilon, self.epsilon_decay, 0.05)
        return loss, reward


if __name__ == "__main__":
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", dest="exp", required=False, type=str)
    parser.add_argument("--gin_config", dest="gin_config", required=False, type=str, default="./config/config_rl.gin")
    parser.add_argument("--checkpoint", default=False, action="store_true")
    args = parser.parse_args()

    # rebind new/overwritten gin parameters
    gin.parse_config_file(args.gin_config)
    if args.exp is not None:
        gin.bind_parameter("%experiment_name", args.exp)

    env = MountainCarEnv()
    trainer = AgentTrainer(env, cbs=[Logger(), Checkpoint(), GradClipping(), TrackerTensorboard()])
    agent = DQNAgent(env)

    # Load from checkpoint
    if args.checkpoint:
        trainer.load_from_checkpoint(agent)
    trainer.train_agent(agent)
