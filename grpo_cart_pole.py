import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import wandb
from tqdm import trange
import os
from dotenv import load_dotenv
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Initializing ActorNetwork...")
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

print("Initializing GRPOTrainer...")
class GRPOTrainer:
    def __init__(self, env, actor, optimizer, 
                 clip_ratio=0.2, beta=0.001, gamma=0.99,
                 epochs=10, batch_size=32, 
                 group_size=200):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.ep_rewards = []
        print(f"GRPOTrainer initialized with clip_ratio={clip_ratio}, beta={beta}, gamma={gamma}, epochs={epochs}, batch_size={batch_size}, group_size={group_size}")

    def _calc_returns(self, rewards, dones):
        print("Calculating returns...")
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(device)

    def collect_rollout(self):
        print("Collecting rollout...")
        states, acts, rews, dones = [], [], [], []
        old_logits = []
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(device)
        ep_rew = 0

        for _ in range(self.group_size):
            with torch.no_grad():
                # Ensure state is always 2D (batched) for DataParallel compatibility
                state_input = state.unsqueeze(0) if state.dim() == 1 else state
                logits = self.actor(state_input)
                logits = logits.squeeze(0)  # Remove batch dimension
                dist = Categorical(logits=logits)
                act = dist.sample()

            next_state, rew, terminated, truncated, _ = self.env.step(act.item())
            done = terminated or truncated
            ep_rew += rew

            states.append(state)
            acts.append(act)
            rews.append(rew)
            dones.append(done)
            old_logits.append(logits)

            state = torch.FloatTensor(next_state).to(device) if not done else torch.FloatTensor(self.env.reset()[0]).to(device)

            if done:
                self.ep_rewards.append(ep_rew)
                ep_rew = 0

        returns = self._calc_returns(rews, dones)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        print("Finished collecting rollout.")
        return (
            torch.stack(states),
            torch.stack(acts),
            advantages,
            torch.stack(old_logits)
        )

    def train(self, total_updates=300):
        print(f"Starting training for {total_updates} updates...")
        self.actor.train()
        for update in trange(total_updates):
            print(f"Update {update+1}/{total_updates}")
            states, actions, advantages, old_logits = self.collect_rollout()

            dataset = torch.utils.data.TensorDataset(states, actions, advantages, old_logits)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            policy_losses = []
            kl_divergences = []

            for _ in range(self.epochs):
                print(f"  Epoch {_+1}/{self.epochs}")
                for batch in loader:
                    print("    Training on batch...")
                    s_batch, a_batch, adv_batch, old_logits_batch = batch

                    new_logits = self.actor(s_batch)
                    old_dist = Categorical(logits=old_logits_batch.detach())
                    new_dist = Categorical(logits=new_logits)

                    logp_new = new_dist.log_prob(a_batch)
                    logp_old = old_dist.log_prob(a_batch).detach()
                    ratio = torch.exp(logp_new - logp_old)


                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv_batch
                    policy_loss = -torch.min(surr1, surr2).mean()


                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

                    loss = policy_loss + self.beta * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    kl_divergences.append(kl.item())


            if self.ep_rewards:
                avg_rew = np.mean(self.ep_rewards[-20:])
                wandb.log({
                    "update": update,
                    "avg_reward": avg_rew,
                    "policy_loss": np.mean(policy_losses),
                    "kl_divergence": np.mean(kl_divergences)
                })
        print("Training complete.")

print("Defining test function...")
def test(env, actor, episodes=5, render=False):
    print(f"Testing for {episodes} episodes...")
    actor.eval()
    for ep in range(episodes):
        print(f"  Test Episode {ep+1}")
        state, _ = env.reset()
        total_rew = 0
        while True:
            if render:
                env.render()
            with torch.no_grad():
                # Ensure state is always 2D (batched) for DataParallel compatibility
                state_tensor = torch.FloatTensor(state).to(device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)  # shape: [1, state_dim]
                logits = actor(state_tensor)
                # Remove batch dimension for action selection
                logits = logits.squeeze(0)
                act = torch.argmax(logits).item()

            state, rew, terminated, truncated, _ = env.step(act)
            total_rew += rew

            if terminated or truncated:
                print(f"Test Episode {ep+1} | Reward: {total_rew}")
                break

print("Defining main function...")
def main(use_data_parallel=True, run_sensitivity=False):
    if run_sensitivity:
        # You can adjust the parameters below as needed
        sensitivity_analysis(
            hidden_sizes_list=[32, 64, 128],
            num_layers_list=[2, 3, 4],
            episodes=300,
            test_episodes=10,
            use_data_parallel=use_data_parallel  # <-- Pass the flag from main
        )
        return
    print("Loading environment variables...")
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        wandb_api_key = wandb_api_key.strip()
        print(f"WANDB_API_KEY repr: {repr(wandb_api_key)}, length: {len(wandb_api_key)}")
    else:
        print("WANDB_API_KEY not found in environment.")
    if wandb_api_key:
        print("Logging in to wandb with API key from environment...")
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not found or empty. wandb logging may fail.")
    print("Initializing wandb...")
    wandb.init(project="grpo-cartpole")

    print("Creating CartPole environment...")
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Creating ActorNetwork and optimizer...")
    actor = ActorNetwork(state_dim, action_dim).to(device)
    # Enable DataParallel for multi-GPU support if flag is set
    print(f"use_data_parallel={use_data_parallel}, torch.cuda.is_available()={torch.cuda.is_available()}, torch.cuda.device_count()={torch.cuda.device_count()}")
    if use_data_parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        actor = nn.DataParallel(actor)
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    print("Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        env=env,
        actor=actor,
        optimizer=optimizer,
        clip_ratio=0.2,
        beta=0.001,
        gamma=0.99,
        epochs=10,
        batch_size=32,
        group_size=200
    )

    print("Starting training...")
    trainer.train(total_updates=1000)

    print("Testing trained policy...")
    test_env = gym.make('CartPole-v1', render_mode='human')
    # If using DataParallel, pass the underlying model to test
    test_actor = actor.module if isinstance(actor, nn.DataParallel) else actor
    test(test_env, test_actor, episodes=30, render=True)
    env.close()

# Flexible ActorNetwork for sensitivity analysis
class FlexibleActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64], activation=nn.Tanh):
        super().__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

# Sensitivity analysis function
def sensitivity_analysis(hidden_sizes_list, num_layers_list, episodes=300, test_episodes=10, use_data_parallel=False):
    # Ensure wandb is initialized for logging
    if not wandb.run:
        wandb.init(project="grpo-cartpole-sensitivity", name="sensitivity_analysis")
    results = []
    for hidden_size in hidden_sizes_list:
        for num_layers in num_layers_list:
            run_name = f"hidden{hidden_size}_layers{num_layers}_dp{use_data_parallel}_{int(time.time())}"
            wandb_run = wandb.init(project="grpo-cartpole-sensitivity", name=run_name, reinit=True)
            print(f"\n--- Training with hidden_size={hidden_size}, num_layers={num_layers}, use_data_parallel={use_data_parallel} ---")
            hidden_sizes = [hidden_size] * num_layers
            env = gym.make('CartPole-v1')
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            actor = FlexibleActorNetwork(state_dim, action_dim, hidden_sizes=hidden_sizes).to(device)
            if use_data_parallel and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for this run")
                actor = nn.DataParallel(actor)
            optimizer = optim.Adam(actor.parameters(), lr=3e-4)
            trainer = GRPOTrainer(
                env=env,
                actor=actor,
                optimizer=optimizer,
                clip_ratio=0.2,
                beta=0.001,
                gamma=0.99,
                epochs=10,
                batch_size=32,
                group_size=200
            )
            trainer.train(total_updates=episodes)
            # Test
            test_env = gym.make('CartPole-v1')
            rewards = []
            test_actor = actor.module if isinstance(actor, nn.DataParallel) else actor
            test_actor.eval()
            for ep in range(test_episodes):
                state, _ = test_env.reset()
                total_rew = 0
                while True:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).to(device)
                        if state_tensor.dim() == 1:
                            state_tensor = state_tensor.unsqueeze(0)
                        logits = test_actor(state_tensor)
                        logits = logits.squeeze(0)
                        act = torch.argmax(logits).item()
                    state, rew, terminated, truncated, _ = test_env.step(act)
                    total_rew += rew
                    if terminated or truncated:
                        break
                rewards.append(total_rew)
            avg_reward = np.mean(rewards)
            print(f"hidden_size={hidden_size}, num_layers={num_layers}, avg_test_reward={avg_reward}")
            wandb_run.log({
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'avg_test_reward': avg_reward,
                'use_data_parallel': use_data_parallel
            })
            wandb_run.finish()
            results.append({
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'avg_test_reward': avg_reward,
                'use_data_parallel': use_data_parallel
            })
    print("\nSensitivity Analysis Results:")
    for r in results:
        print(r)
    return results

# Example usage (uncomment to run):
# sensitivity_analysis(hidden_sizes_list=[32, 64, 128], num_layers_list=[2, 3, 4], episodes=300, test_episodes=10)

if __name__ == "__main__":
    print("Running main...")
    # Set run_sensitivity=True to run the sensitivity analysis, False for normal training
    main(use_data_parallel=True, run_sensitivity=True)