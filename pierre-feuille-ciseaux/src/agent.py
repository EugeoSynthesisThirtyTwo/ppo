import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallAgentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer1 = nn.Linear(3, 20)
        self._layer2 = nn.Linear(20, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)
        x = torch.softmax(x, dim=1)
        return x

class BigAgentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer1 = nn.Linear(3, 64)
        self._layer2 = nn.Linear(64, 64)
        self._layer3 = nn.Linear(64, 64)
        self._layer4 = nn.Linear(64, 64)
        self._layer5 = nn.Linear(64, 3)

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
        x = torch.softmax(x, dim=1)
        return x

class Agent(BigAgentModel):
    def predict_action(self, state: torch.Tensor, exploration: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        exploration should be between 0 and 1
        return original_action, action_with_exploration
        """
        original_action = self(state)

        with torch.no_grad():
            noise = 2 * torch.rand_like(original_action) - 1
            action_with_exploration = original_action * (1 + exploration * noise)
        
        return original_action, action_with_exploration

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    @staticmethod
    def load(path: str):
        agent = Agent()
        agent.load_state_dict(torch.load(path, weights_only=True))
        return agent
        