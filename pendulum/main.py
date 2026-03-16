import torch
from src.window.game_pendulum_on_rail import GamePendulumOnRail
from src.ppo.train import TrainWithCritic, TrainWithoutCritic

train = TrainWithCritic()
train.train()
# train.train_no_critic()
# train.train_simultaneously()

# GamePendulumOnRail().loop_forever()