import os
import gymnasium as gym
import time
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange

torch.backends.cudnn.benchmark = True

class ReplayBuffer:
    def __init__(self, capacity, state_shape=(4, 84, 84)):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

    def __len__(self):
        return self.size

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.head(x)

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_action(env, model, epsilon, obs) -> int:
    if np.random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():
        obs_np = np.array(obs, dtype=np.uint8)
        state_t = torch.tensor(obs_np, device=device).unsqueeze(0).float()
        q_values = model(state_t)
        return int(q_values.argmax(dim=1).item())

def evaluate_agent(model, n_episodes=1):
    print(f"\n--- Starting Evaluation ({n_episodes} episode) ---")
    eval_env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=1)
    eval_env = gym.wrappers.AtariPreprocessing(eval_env, frame_skip=4, grayscale_obs=True, screen_size=84)
    eval_env = gym.wrappers.FrameStackObservation(eval_env, 4)

    try:
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = get_action(eval_env, model, epsilon=0, obs=obs)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"Evaluation Score: {total_reward}")
    except Exception as e:
        print(f"Evaluation interrupted or window closed: {e}")
    finally:
        eval_env.close()
    print("--- Evaluation Finished ---\n")

def main():
    os.makedirs("checkpoints", exist_ok=True)
    env = gym.make("ALE/Pong-v5", render_mode=None, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84)
    env = gym.wrappers.FrameStackObservation(env, 4)

    n_actions = env.action_space.n
    q_network = DQN(n_actions).to(device)
    target_network = DQN(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=0.0001)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_duration = 1_000_000

    batch_size = 256
    gamma = 0.99
    N_episodes = 10000
    target_network_frequency = 20000
    render_frequency = 1000
    replay_buffer_size = 250000
    train_frequency = 4
    warmup_steps = 1000

    memory = ReplayBuffer(replay_buffer_size)
    total_steps = 0
    epsilon = epsilon_start
    print(f"DQN training started on {device}. Linear epsilon decay over {epsilon_duration} steps.")

    try:
        pbar = trange(N_episodes, desc="Episodes")
        for i in pbar:
            if i > 0 and i % render_frequency == 0:
                evaluate_agent(q_network)

            observation, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_duration)
                action = get_action(env, q_network, epsilon, observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1

                memory.push(observation, action, reward, next_observation, terminated or truncated)
                observation = next_observation

                if total_steps > warmup_steps and total_steps % train_frequency == 0:
                    states_np, actions_np, rewards_np, next_states_np, dones_np = memory.sample(batch_size)

                    states = torch.tensor(states_np, device=device).float()
                    actions = torch.tensor(actions_np, dtype=torch.long, device=device).unsqueeze(1)
                    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                    next_states = torch.tensor(next_states_np, device=device).float()
                    dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

                    current_q = q_network(states).gather(1, actions).squeeze(1)
                    with torch.no_grad():
                        next_q = target_network(next_states).max(dim=1)[0]
                        target_q = rewards + (gamma * next_q * (1 - dones))

                    loss = F.mse_loss(current_q, target_q)
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                if total_steps % target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())

            pbar.set_postfix(reward=total_reward, eps=f"{epsilon:.2f}", steps=total_steps)

            if i % 100 == 0:
                print(f"Episode {i} finished. Total Reward: {total_reward}. Epsilon: {epsilon:.2f}")
            if i > 0 and i % 1000 == 0:
                checkpoint_path = f"checkpoints/dqn_pong_episode_{i}.pth"
                torch.save({
                    "episode": i,
                    "model_state_dict": q_network.state_dict(),
                    "target_state_dict": target_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "total_steps": total_steps
                }, checkpoint_path)
                print(f"Model saved at episode {i} -> {checkpoint_path}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()