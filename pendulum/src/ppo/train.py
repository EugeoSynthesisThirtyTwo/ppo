import time

import torch
import torch.optim as optim
import numpy as np

from src.ppo.ui import UiPendulumOnRail
from src.ppo.agent import Agent
from src.ppo.critic import Critic
from src.ppo.environment import EnvironmentPendulumOnRailWithFriction, Episode
from src.debug.profiling import profile_context
from src.debug.logger import logger

class Train:
    def __init__(self):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._half_life_gamma = 40

        self._simulation_duration = 40
        self._dt = 0.02
        self._simulation_update_per_action = 1
        self._actions_per_ui_refresh = 4
        self._ui_enable_every_n_epoch = 5
        self._randomized = False
        self._force_multiplier = 100

        self._epochs = 10000
        self._batch_size = 1024
        self._agent_learning_rate = 0.001
        self._critic_learning_rate = 0.001
        self._updates_critic_per_agent = 10
        self._device = "cuda:0"

        self._ui = UiPendulumOnRail()
        self._agent = Agent(device=self._device)
        self._critic = Critic(device=self._device, action_in_input=False)
        # self._agent = Agent.load("latest_agent.bin", device=self._device)
        # self._critic = Critic.load("latest_critic.bin", device=self._device, action_in_input=False)
    
    def preprocess_state(self, state: torch.Tensor, time_between_0_and_1: float) -> torch.Tensor:
        theta_pos = state[:, 2:3]
        cos_theta = torch.cos(theta_pos)
        sin_theta = torch.sin(theta_pos)
        t = time_between_0_and_1 * torch.ones_like(theta_pos, device=self._device)
        return torch.cat((state[:, :2], theta_pos, cos_theta, sin_theta, state[:, 3:], t), dim=1)
    
    def epoch(self, steps: int, ui_update: bool) -> Episode:
        self._environment = EnvironmentPendulumOnRailWithFriction(batch_size=self._batch_size, device=self._device, dt=self._dt, randomized_state=self._randomized)

        for step in range(steps):
            with torch.no_grad():
                state = self._environment.get_state()
                state = self.preprocess_state(state, step / steps)
            
            mean, std, action = self._agent.predict_action(state, self._force_multiplier)
            
            with torch.no_grad():
                self._environment.action(state, mean, std, action, self._simulation_update_per_action)

            if not self._ui.is_running():
                exit(1)
            
            if (ui_update and step % self._actions_per_ui_refresh == 0) or step == 0:
                self._ui.next_frame(self._environment.get_pendulum(), self._dt / self._actions_per_ui_refresh, index_in_batch=0)

        with torch.no_grad():
            episode = self._environment.end(self._half_life_gamma)
        
        return episode

class TrainWithoutCritic(Train):
    def train(self):
        steps = int(self._simulation_duration / self._dt)
        optimizer_agent = optim.Adam(self._agent.parameters(), lr=self._agent_learning_rate)

        for epoch in range(self._epochs):
            logger.info(f"epoch {epoch}")

            with profile_context(f"epoch {epoch}"):
                optimizer_agent.zero_grad()
                ui_update = (epoch % self._ui_enable_every_n_epoch == 0)
                episode = self.epoch(steps, ui_update)
                batch_state = episode.get_batch_key("state")
                batch_mean = episode.get_batch_key("mean")
                batch_std = episode.get_batch_key("std")
                batch_action = episode.get_batch_key("action")
                batch_diminished_reward = episode.get_batch_key("diminished_reward")
                print(f"average diminished reward: {torch.mean(batch_diminished_reward)}")
            
                with torch.no_grad():
                    advantage = (batch_diminished_reward - batch_diminished_reward.mean()) / (batch_diminished_reward.max() - batch_diminished_reward.min() + 1e-4)
                    distrib = torch.distributions.Normal(batch_mean, batch_std)
                
                log_probs = distrib.log_prob(batch_action)
                loss_agent = -(advantage * log_probs).mean()
                loss_agent.backward()
                optimizer_agent.step()
                self._agent.save("latest_agent.bin")

        self._agent.save("trained_agent.bin")

class TrainWithCritic(Train):
    def train(self):
        steps = int(self._simulation_duration / self._dt)
        optimizer_agent = optim.Adam(self._agent.parameters(), lr=self._agent_learning_rate)
        optimizer_critic = optim.Adam(self._critic.parameters(), lr=self._critic_learning_rate)

        for epoch in range(self._epochs):
            logger.info(f"epoch {epoch}")

            with profile_context(f"epoch {epoch}"):
                optimizer_agent.zero_grad()
                ui_update = (epoch % self._ui_enable_every_n_epoch == 0)

                episode = self.epoch(steps, ui_update)
                
                batch_state = episode.get_batch_key("state")
                batch_mean = episode.get_batch_key("mean")
                batch_std = episode.get_batch_key("std")
                batch_action = episode.get_batch_key("action")
                batch_diminished_reward = episode.get_batch_key("diminished_reward")
                logger.info(f"average diminished reward: {torch.mean(batch_diminished_reward)}")
    
                # critic
                with torch.no_grad():
                    batch_state_reshaped = torch.reshape(batch_state, (batch_state.shape[0] * batch_state.shape[1], batch_state.shape[2]))
                    batch_action_reshaped = torch.reshape(batch_action, (batch_action.shape[0] * batch_action.shape[1], batch_action.shape[2]))
                
                for _ in range(self._updates_critic_per_agent):
                    optimizer_critic.zero_grad()
                    value_reshaped = self._critic.predict_diminished_reward(batch_state_reshaped.detach(), batch_action_reshaped.detach())
                    value = torch.reshape(value_reshaped, batch_diminished_reward.shape)
                    loss_critic = torch.mean(((value - batch_diminished_reward) / (batch_diminished_reward.max() - batch_diminished_reward.min() + 1e-4)) ** 2)
                    loss_critic.backward()
                    optimizer_critic.step()

                value_reshaped = self._critic.predict_diminished_reward(batch_state_reshaped.detach(), batch_action_reshaped.detach())
                value = torch.reshape(value_reshaped, batch_diminished_reward.shape)
                loss_critic = torch.mean(((value - batch_diminished_reward) / (batch_diminished_reward.max() - batch_diminished_reward.min() + 1e-4)) ** 2)
                logger.info(f"loss critic: {loss_critic}")
                
                # agent
                with torch.no_grad():   
                    value = value.detach()
                    advantage = (batch_diminished_reward - value) / (value.max() - value.min() + 1e-4)
                    distrib = torch.distributions.Normal(batch_mean, batch_std)
                
                log_probs = distrib.log_prob(batch_action)
                loss_agent = -(advantage * log_probs).mean()
                loss_agent.backward()
                optimizer_agent.step()
                self._agent.save("latest_agent.bin")
                self._critic.save("latest_critic.bin")

        self._agent.save("trained_agent.bin")
        self._critic.save("trained_critic.bin")

class TrainWithCriticPPO(Train):
    def train(self):
        steps = int(self._simulation_duration / self._dt)
        optimizer_agent = optim.Adam(self._agent.parameters(), lr=self._agent_learning_rate)
        optimizer_critic = optim.Adam(self._critic.parameters(), lr=self._critic_learning_rate)

        for epoch in range(self._epochs):
            logger.info(f"epoch {epoch}")

            with profile_context(f"epoch {epoch}"):
                optimizer_agent.zero_grad()
                ui_update = (epoch % self._ui_enable_every_n_epoch == 0)

                episode = self.epoch(steps, ui_update)
                
                batch_state = episode.get_batch_key("state")
                batch_mean = episode.get_batch_key("mean")
                batch_std = episode.get_batch_key("std")
                batch_action = episode.get_batch_key("action")
                batch_diminished_reward = episode.get_batch_key("diminished_reward")
                logger.info(f"average diminished reward: {torch.mean(batch_diminished_reward)}")
    
                # critic
                with torch.no_grad():
                    batch_state_reshaped = torch.reshape(batch_state, (batch_state.shape[0] * batch_state.shape[1], batch_state.shape[2]))
                    batch_action_reshaped = torch.reshape(batch_action, (batch_action.shape[0] * batch_action.shape[1], batch_action.shape[2]))
                
                for _ in range(self._updates_critic_per_agent):
                    optimizer_critic.zero_grad()
                    value_reshaped = self._critic.predict_diminished_reward(batch_state_reshaped.detach(), batch_action_reshaped.detach())
                    value = torch.reshape(value_reshaped, batch_diminished_reward.shape)
                    loss_critic = torch.mean(((value - batch_diminished_reward) / (batch_diminished_reward.max() - batch_diminished_reward.min() + 1e-4)) ** 2)
                    loss_critic.backward()
                    optimizer_critic.step()

                value_reshaped = self._critic.predict_diminished_reward(batch_state_reshaped.detach(), batch_action_reshaped.detach())
                value = torch.reshape(value_reshaped, batch_diminished_reward.shape)
                loss_critic = torch.mean(((value - batch_diminished_reward) / (batch_diminished_reward.max() - batch_diminished_reward.min() + 1e-4)) ** 2)
                logger.info(f"loss critic: {loss_critic}")
                
                # agent
                with torch.no_grad():   
                    value = value.detach()
                    advantage = (batch_diminished_reward - value) / (value.max() - value.min() + 1e-4)
                    distrib = torch.distributions.Normal(batch_mean, batch_std)
                
                log_probs = distrib.log_prob(batch_action)
                loss_agent = -(advantage * log_probs).mean()
                loss_agent.backward()
                optimizer_agent.step()
                self._agent.save("latest_agent.bin")
                self._critic.save("latest_critic.bin")

        self._agent.save("trained_agent.bin")
        self._critic.save("trained_critic.bin")