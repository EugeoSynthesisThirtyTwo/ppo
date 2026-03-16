import torch
import numpy as np

SAR = dict[str, torch.Tensor]

class Episode:
    def __init__(self):
        self._episode: list[SAR] = []
    
    def append(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, **kw_args):
        sar = {
            "state": state,
            "action": action,
            "reward": reward,
            "diminished_reward": None
        } | kw_args
        self._episode.append(sar)

    def end_episode(self, gamma: float):
        coeff = 0

        with torch.no_grad():
            diminished_reward = torch.zeros_like(self._episode[0]["reward"])

            for index in range(len(self._episode) - 1, -1, -1):
                sar = self._episode[index]
                coeff = 1 + gamma * coeff
                diminished_reward = sar["reward"] + gamma * diminished_reward
                sar["diminished_reward"] = diminished_reward / coeff
            
            for index in range(len(self._episode) - 1, -1, -1):
                sar = self._episode[index]
                delta = sar["reward"] - 
    
    def get_batch_key(self, key: str) -> torch.Tensor:
        batch = [sar[key] for sar in self._episode]
        return torch.stack(batch, dim=0)
    
    def sample(self, batch_size: int = -1):
        if batch_size == -1:
            return self
        
        indices = np.random.choice(len(self._episode), batch_size)
        return [self._episode[i] for i in indices]