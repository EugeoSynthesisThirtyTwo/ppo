import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =============================================================
# Minimal N-armed Bandit as a discrete-action RL environment
# -------------------------------------------------------------
# - action space: {0, 1, ..., n_actions-1}
# - state: a dummy zero vector (bandits have no informative state)
# - reward: Bernoulli with action-specific success probability
# =============================================================

class BanditEnv:
    def __init__(self, n_actions: int = 3, drift: float = 0.0, seed: int = 0):
        self.n_actions = n_actions
        self.state_dim = 4  # dummy state; PPO expects some input
        g = torch.Generator().manual_seed(seed)
        self.p = torch.rand(n_actions, generator=g).tolist()
        self.drift = drift
        self.step_count = 0

    def reset(self):
        # return a (state, info) tuple matching Gym-like API
        self.step_count = 0
        s = torch.zeros(self.state_dim)
        return s, {}

    def step(self, a: int):
        assert 0 <= a < self.n_actions
        # Bernoulli reward with action-dependent probability
        r = 1.0 if random.random() < self.p[a] else 0.0
        self.step_count += 1

        # Optional: very small nonstationarity
        if self.drift != 0.0 and self.step_count % 100 == 0:
            j = random.randrange(self.n_actions)
            self.p[j] = min(0.99, max(0.01, self.p[j] + random.uniform(-self.drift, self.drift)))

        # bandit has no terminal state; we create short episodes
        terminated = False
        truncated = False
        s_next = torch.zeros(self.state_dim)
        info = {"p": self.p.copy()}
        return s_next, float(r), terminated, truncated, info


# =============================================================
# Policy & Value Networks (shared body, separate heads)
# =============================================================

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)   # outputs logits
        self.critic = nn.Linear(hidden, 1)          # outputs V(s)

    def forward(self, s: torch.Tensor):
        z = self.body(s)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    def act(self, s: torch.Tensor):
        logits, v = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v

    def log_prob(self, s: torch.Tensor, a: torch.Tensor):
        logits, _ = self.forward(s)
        dist = Categorical(logits=logits)
        return dist.log_prob(a)


# =============================================================
# Utilities: GAE, discounted returns
# =============================================================

def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float,
                lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given per-step rewards r_t and value predictions V(s_t), compute
    advantages Â_t and returns R_t using Generalized Advantage Estimation.

    Args:
        rewards: shape [T]
        values:  shape [T + 1]  (bootstrap value at T included)
        dones:   shape [T]  (True if episode ended after t)
    Returns:
        advantages [T], returns [T]
    """
    T = rewards.size(0)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns


# =============================================================
# PPO Trainer
# =============================================================

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95           # GAE(lambda)
    clip_eps: float = 0.2       # PPO clip range
    ent_coef: float = 0.01      # entropy bonus weight
    vf_coef: float = 0.5        # value loss weight
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    rollout_steps: int = 256    # T
    minibatch_size: int = 64
    update_epochs: int = 4
    total_updates: int = 500
    device: str = "cpu"


class PPO:
    def __init__(self, state_dim: int, n_actions: int, cfg: PPOConfig):
        self.cfg = cfg
        self.net = ActorCritic(state_dim, n_actions).to(cfg.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def collect_rollout(self, env: BanditEnv) -> dict:
        T = self.cfg.rollout_steps
        device = self.cfg.device

        s, _ = env.reset()
        s = s.to(device)

        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        for _ in range(T):
            a, logp, v = self.net.act(s)
            s_next, r, terminated, truncated, _ = env.step(int(a.item()))

            states.append(s)
            actions.append(a)
            logps.append(logp)
            rewards.append(torch.tensor(r, device=device))
            dones.append(torch.tensor(float(terminated or truncated), device=device))
            values.append(v)

            s = s_next.to(device)

        # bootstrap value V(s_T)
        last_v = self.net.forward(s)[1]
        values = torch.stack(values + [last_v])  # shape [T+1]

        batch = {
            "states": torch.stack(states),           # [T, state_dim]
            "actions": torch.stack(actions),         # [T]
            "logps": torch.stack(logps),             # [T]
            "rewards": torch.stack(rewards),         # [T]
            "dones": torch.stack(dones),             # [T]
            "values": values,                        # [T+1]
        }
        return batch

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        device = cfg.device

        # ----- Compute advantages and returns (math Steps 2–4)
        advantages, returns = compute_gae(
            batch["rewards"], batch["values"], batch["dones"], cfg.gamma, cfg.lam
        )
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare flat tensors for minibatching
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        old_logps = batch["logps"].to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        # Indices for shuffling
        N = states.size(0)
        idxs = torch.arange(N)

        # Metrics to log
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0

        for _ in range(cfg.update_epochs):
            perm = idxs[torch.randperm(N)]
            for start in range(0, N, cfg.minibatch_size):
                mb_idx = perm[start:start + cfg.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_advs = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # ----- Step 5: new log-probs and values
                logits, values = self.net.forward(mb_states)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ----- Step 6: Clipped policy loss (surrogate)
                # r_t(θ) = exp(log π_θ(a|s) - log π_{θ_old}(a|s))
                ratio = (new_logps - mb_old_logps).exp()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # ----- Step 7: Value loss
                value_loss = 0.5 * (mb_returns - values.squeeze(-1)).pow(2).mean()

                # ----- Step 8: Entropy bonus
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                # Optimization step
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()

        steps = math.ceil(N / cfg.minibatch_size) * cfg.update_epochs
        log = {
            "policy_loss": policy_loss_epoch / steps,
            "value_loss": value_loss_epoch / steps,
            "entropy": entropy_epoch / steps,
        }
        return log


def train_demo(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)

    # ----- Create bandit and PPO agent
    env = BanditEnv(n_actions=3, drift=0.00, seed=seed)
    cfg = PPOConfig(total_updates=200, rollout_steps=256, device="cpu")
    agent = PPO(state_dim=env.state_dim, n_actions=env.n_actions, cfg=cfg)

    # Track moving average of reward to see improvement
    moving_avg = None

    for update in range(cfg.total_updates):
        batch = agent.collect_rollout(env)
        # Average reward over the rollout (not discounted) for logging only
        avg_reward = batch["rewards"].mean().item()
        moving_avg = avg_reward if moving_avg is None else 0.95 * moving_avg + 0.05 * avg_reward

        stats = agent.update(batch)

        if (update + 1) % 10 == 0:
            print(
                f"upd {update+1:4d} | avg_r={avg_reward:.3f} ema_r={moving_avg:.3f} "
                f"| pi_loss={stats['policy_loss']:.3f} v_loss={stats['value_loss']:.3f} H={stats['entropy']:.3f}"
            )

    # After training, print learned action probabilities
    with torch.no_grad():
        s, _ = env.reset()
        logits, _ = agent.net.forward(s.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        print("\nLearned action probs:", probs.tolist())
        print("True bandit success probs:", env.p)


if __name__ == "__main__":
    train_demo(seed=0)
