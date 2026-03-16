import torch
from math import exp, log

class StateActionReward:
    def __init__(self, state: torch.Tensor, original_action: torch.Tensor, action_with_exploration: torch.Tensor, reward: torch.Tensor):
        self._state = state
        self._original_action = original_action
        self._action_with_exploration = action_with_exploration
        self._reward = reward
        self._diminished_reward = 0
    
    def get_state(self) -> torch.Tensor:
        return self._state
    
    def set_state(self, state: torch.Tensor):
        self._state = state
    
    def get_original_action(self) -> torch.Tensor:
        return self._original_action
    
    def set_original_action(self, original_action: torch.Tensor):
        self._original_action = original_action
    
    def get_action_with_exploration(self) -> torch.Tensor:
        return self._action_with_exploration
    
    def set_action_with_exploration(self, action_with_exploration: torch.Tensor):
        self._action_with_exploration = action_with_exploration
    
    def get_reward(self) -> torch.Tensor:
        return self._reward
    
    def set_reward(self, reward: torch.Tensor):
        self._reward = reward
    
    def get_diminished_reward(self) -> torch.Tensor:
        return self._diminished_reward
    
    def set_diminished_reward(self, diminished_reward: torch.Tensor):
        self._diminished_reward = diminished_reward

class Episode:
    def __init__(self):
        self._episode: list[StateActionReward] = []
    
    def append(self, state: torch.Tensor, original_action: torch.Tensor, action_with_exploration: torch.Tensor, reward: torch.Tensor):
        sar = StateActionReward(state, original_action, action_with_exploration, reward)
        self._episode.append(sar)

    def end_episode(self, gamma: float):
        coeff = 0

        with torch.no_grad():
            diminished_reward = torch.zeros_like(self._episode[0].get_reward())

            for index in range(len(self._episode) - 1, -1, -1):
                sar = self._episode[index]
                coeff = 1 + gamma * coeff
                diminished_reward = sar.get_reward() + gamma * diminished_reward
                sar.set_diminished_reward(diminished_reward / coeff)
    
    def get_batch_state(self) -> torch.Tensor:
        batch = [sar.get_state() for sar in self._episode]
        return torch.cat(batch, dim=0)
    
    def get_batch_original_action(self) -> torch.Tensor:
        batch = [sar.get_original_action() for sar in self._episode]
        return torch.cat(batch, dim=0)
    
    def get_batch_action_with_exploration(self) -> torch.Tensor:
        batch = [sar.get_action_with_exploration() for sar in self._episode]
        return torch.cat(batch, dim=0)
    
    def get_batch_reward(self) -> torch.Tensor:
        batch = [sar.get_reward() for sar in self._episode]
        return torch.cat(batch, dim=0).unsqueeze(1)
    
    def get_batch_diminished_reward(self) -> torch.Tensor:
        batch = [sar.get_diminished_reward() for sar in self._episode]
        return torch.cat(batch, dim=0).unsqueeze(1)

class EnvironmentPierreFeuilleCiseaux:
    def __init__(self, batch_size: int, device: str, dt: float, randomized_state: bool):
        self._episode = Episode()
        self._pfs = PierreFeuilleCiseaux()
        self._dt = dt
    
    def _get_reward(self, action: torch.Tensor, prev_state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pass
    
    def action(self, state: torch.Tensor, original_action: torch.Tensor, action_with_exploration: torch.Tensor):
        with torch.no_grad():
            prev_state = self._pendulum.get_state()
            self._pendulum.next_state(self._dt, action_with_exploration)
            new_state = self._pendulum.get_state()
            reward = self._get_reward(action_with_exploration, prev_state, new_state)
            self._episode.append(state, original_action, action_with_exploration, reward)
    
    def end(self, half_life: float) -> Episode:
        """
        half_life is the half life (obviously) of the exponential diminished returns in seconds. Can also be 0 or float("inf").
        """
        if half_life == 0:
            gamma = 0
        elif half_life == float("inf"):
            gamma = 1
        else:
            gamma = exp(-log(2) * self._dt / half_life)

        self._episode.end_episode(gamma)
        return self._episode
    
    def get_pendulum(self) -> PendulumOnRailWithFriction:
        return self._pendulum
    
    def get_state(self) -> torch.Tensor:
        return self._pendulum.get_state()
