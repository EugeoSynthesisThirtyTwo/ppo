import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCriticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer1 = nn.Linear(8, 20)
        self._layer2 = nn.Linear(20, 1)
        self._layer1 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer1(x)
        # x = F.relu(x)
        # x = self._layer2(x)
        return x

class BigCriticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer1 = nn.Linear(8, 64)
        self._layer2 = nn.Linear(64, 64)
        self._layer3 = nn.Linear(64, 64)
        self._layer4 = nn.Linear(64, 64)
        self._layer5 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)
        x = F.relu(x)
        x = self._layer3(x)
        x = F.relu(x)
        x = self._layer4(x)
        x = F.relu(x)
        x = self._layer5(x)
        return x

class Critic:
    def __init__(self, big: bool = False):
        if big:
            self._model = BigCriticModel()
        else:
            self._model = SmallCriticModel()
    
    def predict_diminished_reward(self, state: torch.Tensor, action_with_exploration: torch.Tensor) -> torch.Tensor:
        """
        exploration should be between 0 and 1
        """
        input = torch.cat((state, action_with_exploration), dim=1)
        diminished_reward = self._model(input)
        return diminished_reward

    def parameters(self):
        return self._model.parameters()

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)
    
    @staticmethod
    def load(path: str):
        agent = Critic()
        agent._model.load_state_dict(torch.load(path, weights_only=True))
        return agent
    
    def to(self, device: str):
        return self._model.to(device)