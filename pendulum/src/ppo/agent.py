import torch
import torch.nn as nn
import torch.nn.functional as F

class BigContinuousAgentModel(nn.Module):
    def __init__(self, n_continuous_actions: int) -> None:
        super().__init__()
        self._layer1 = nn.Linear(7, 64)
        self._layer2 = nn.Linear(64, 64)
        self._layer3 = nn.Linear(64, 64)
        self._layer4 = nn.Linear(64, 64)

        self._mean_layer = nn.Linear(64, n_continuous_actions)
        self._log_std_layer = nn.Linear(64, n_continuous_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)
        x = F.relu(x)
        x = self._layer3(x)
        x = F.relu(x)
        x = self._layer4(x)
        x = F.relu(x)

        mean = self._mean_layer(x)
        log_std = self._log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std

class Agent:
    def __init__(self, device: str):
        self._model = BigContinuousAgentModel(1).to(device)
    
    def predict_action(self, state: torch.Tensor, force_multiplier: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self._model(state)
        mean = mean * force_multiplier
        std = std * force_multiplier
        
        with torch.no_grad():
            distrib = torch.distributions.Normal(mean, std)
            action = distrib.sample()

        return mean, std, action

    def parameters(self):
        return self._model.parameters()

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)
    
    @staticmethod
    def load(path: str, device: str):
        agent = Agent(device)
        agent._model.load_state_dict(torch.load(path, weights_only=True))
        return agent
    
    def to(self, device: str):
        return self._model.to(device)