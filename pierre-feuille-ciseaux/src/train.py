import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agent import Agent

class Train:
    def __init__(self):
        self._epochs = 1000
        self._batch_size = 1024
        self._learning_rate = 0.001
        self._device = "cuda:0"
        self._dtype = torch.float32

        self._exploration = 0.2

        self._big_model = False
        self._agent = Agent().to(device=self._device, dtype=self._dtype)

        self._loss_over_time = []

        self._experiment_name = "experiment 2"
        self._experiment_description = f"""
        {self._agent.__class__.__name__}
        {self._epochs} epochs
        {self._batch_size} batch size
        {self._learning_rate} learning rate

        Simple training with labelisation.
        """
    
    def update_plot(self, epoch: int, loss: torch.Tensor):
        self._loss_over_time.append((epoch, loss.cpu().item()))
    
    def save_experiment(self):
        path = f"runs/{self._experiment_name}"
        os.makedirs(path, exist_ok=True)
        self._agent.save(f"{path}/model.pth")
        
        with open(f"{path}/description.txt", "w") as f:
            description = self._experiment_description.splitlines()
            while description[0] == "":
                del description[0]
            
            margin = 0
            while description[0][margin] == " ":
                margin += 1
            
            for i in range(len(description)):
                description[i] = description[i][margin:] + "\n"
            
            for i in range(len(description) - 1, -1, -1):
                if description[i] == "\n":
                    del description[i]
                else:
                    break
            
            description[-1] = description[-1][:-2]
            f.writelines(description)
        
        epochs = [epoch for epoch, loss in self._loss_over_time]
        losses = [loss for epoch, loss in self._loss_over_time]
        plt.plot(epochs, losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.savefig(f"{path}/loss.png", dpi=300)
    
    def generate_training_batch(self, batch_size) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            input_indexes = torch.randint(0, 3, (batch_size,))
            batch_indexes = torch.arange(batch_size)

            input_batch = torch.zeros((batch_size, 3), dtype=self._dtype)
            input_batch[batch_indexes, input_indexes] = 1

            label_indexes = (input_indexes + 1) % 3
            label_batch = torch.zeros((batch_size, 3), dtype=self._dtype)
            label_batch[batch_indexes, label_indexes] = 1

            return input_batch.to(self._device), label_batch.to(self._device)
    
    def generate_test_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            input_batch = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=self._dtype)
            label_batch = torch.tensor([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ], dtype=self._dtype)
            return input_batch.to(self._device), label_batch.to(self._device)
    
    def test(self) -> torch.Tensor:
        input_batch, label_batch = self.generate_test_batch()
        output_batch = self._agent(input_batch)
        loss = torch.mean(torch.square(output_batch - label_batch))
        return loss
    
    def train_agent_basic(self):
        optimizer = torch.optim.Adam(self._agent.parameters(), lr=self._learning_rate)

        for epoch in range(self._epochs):
            optimizer.zero_grad()
            input_batch, label_batch = self.generate_training_batch(self._batch_size)
            original_action, action_with_exploration = self._agent.predict_action(input_batch, self._exploration)
            loss_policy = torch.mean(torch.square(original_action - label_batch))
            loss = self.test()
            loss_policy.backward()
            optimizer.step()
            self.update_plot(epoch, loss)

            if epoch % max(1, self._epochs // 10) == 0:
                print(f"epoch {epoch}: {loss_policy=}, loss={loss}")

        input_batch, label_batch = self.generate_test_batch()
        print(f"Test:\n{input_batch=}\noutput_batch={self._agent(input_batch)}")
        self.save_experiment()
    
    def train_agent_ppo(self):
        optimizer = torch.optim.Adam(self._agent.parameters(), lr=self._learning_rate)

        for epoch in range(self._epochs):
            optimizer.zero_grad()
            input_batch, label_batch = self.generate_training_batch(self._batch_size)

            probs = self._agent(input_batch)
            
            with torch.no_grad():
                distrib = torch.distributions.Categorical(probs)
                action_indexes = distrib.sample()
                
                action = torch.zeros_like(probs, dtype=self._dtype)
                batch_indexes = torch.arange(input_batch.shape[0])
                action[batch_indexes, action_indexes] = 1
                reward = torch.sum(1 - torch.square(action - label_batch), dim=1)
                advantage = (reward - reward.mean()) / (reward.std() + 1e-8)

            distrib = torch.distributions.Categorical(probs)
            log_probs = distrib.log_prob(action_indexes)
            loss_policy = -(log_probs * advantage).mean()
            loss = self.test()

            loss_policy.backward()
            optimizer.step()
            self.update_plot(epoch, loss)

            if epoch % max(1, self._epochs // 10) == 0:
                print(f"epoch {epoch}: loss={loss}")
                print(f"{advantage=}")

            # if epoch % max(1, self._epochs // 10) == 0:
            #     print(f"epoch {epoch}: loss_policy={loss_policy.item()}, loss={loss}")

        input_batch, label_batch = self.generate_test_batch()
        print(f"Test:\ninput:\n{input_batch}\noutput:\n{self._agent(input_batch)}")
        self.save_experiment()
    
if __name__ == "__main__":
    torch.manual_seed(43)
    train = Train()
    train.train_agent_ppo()