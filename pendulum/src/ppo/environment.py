import torch
from math import exp, log

from src.pendulum.pendulum_on_rail import PendulumOnRailWithFriction
from src.ppo.episode import Episode

class EnvironmentPendulumOnRailWithFriction:
    def __init__(self, batch_size: int, device: str, dt: float, randomized_state: bool):
        self._episode = Episode()
        self._pendulum = PendulumOnRailWithFriction(1, 50, 0.5, randomized_state=randomized_state, batch_size=batch_size, device=device)
        self._dt = dt

    def _get_reward(self, new_state: torch.Tensor, bounce: torch.Tensor, dt: float) -> torch.Tensor:
        with torch.no_grad():
            x0 = new_state[:, 0]
            spd0 = new_state[:, 1]
            theta = new_state[:, 2]
            spd_theta = new_state[:, 3]
            etendu = self._pendulum.get_xlim()
            etendu = etendu[1] - etendu[0]
            reward = (1 + torch.exp(-0.5 * torch.square(spd_theta))) * torch.square(1 - torch.cos(theta))
            return reward * dt - 4 * bounce
    
    def action(self, state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor, number_of_simulation_update_per_action: int):
        with torch.no_grad():
            total_reward = torch.zeros((action.shape[0], 1), device=state.device)
            real_dt = self._dt / number_of_simulation_update_per_action
            
            for _ in range(number_of_simulation_update_per_action):
                bounce = self._pendulum.next_state(real_dt, action)
                new_state = self._pendulum.get_state()
                reward = self._get_reward(new_state, bounce, real_dt)
                total_reward = total_reward + reward.unsqueeze(-1)
            
            self._episode.append(state.detach(), action.detach(), total_reward.detach(), mean=mean, std=std)
    
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
