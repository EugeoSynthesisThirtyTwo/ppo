import torch
import torch.nn as nn
import torch.nn.functional as F

class BigCriticModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._layer1 = nn.Linear(input_size, 64)
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
    def __init__(self, device: str, action_in_input: bool):
        self._action_in_input = action_in_input
        self._model = BigCriticModel(8 if action_in_input else 7).to(device)
    
    def predict_diminished_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._action_in_input:
            input = torch.cat((state, action), dim=1)
        else:
            input = state
        
        diminished_reward = self._model(input)
        return diminished_reward

    def parameters(self):
        return self._model.parameters()

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)
    
    @staticmethod
    def load(path: str, device: str, action_in_input: bool):
        agent = Critic(device, action_in_input)
        agent._model.load_state_dict(torch.load(path, weights_only=True))
        return agent
    
    def to(self, device: str):
        return self._model.to(device)