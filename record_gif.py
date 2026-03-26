import os
import gymnasium as gym
import ale_py
import torch
import numpy as np
import imageio
from dql import DQN

def get_action(model, obs, device) -> int:
    """Choose the best action from the model."""
    with torch.no_grad():
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t)
        return int(q_values.argmax(dim=1).item())

def main():
    checkpoints_dir = "checkpoints"
    gifs_dir = "gifs"
    os.makedirs(gifs_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    gym.register_envs(ale_py)
    # Use rgb_array for recording
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    # Preprocessing wrappers (same as play_pong.py)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84)
    env = gym.wrappers.FrameStackObservation(env, 4)

    n_actions = env.action_space.n
    model = DQN(n_actions).to(device)

    # Find all checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    checkpoint_files.sort() # Sort to process in order

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoints_dir}")
        env.close()
        return

    for checkpoint_name in checkpoint_files:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        output_gif = os.path.join(gifs_dir, checkpoint_name.replace(".pth", ".gif"))
        
        print(f"\nProcessing: {checkpoint_name}")
        
        try:
            # Load the weights
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"Weights loaded successfully from {checkpoint_name}")
        except Exception as e:
            print(f"Error loading weights from {checkpoint_name}: {e}")
            continue

        print(f"Playing one episode to record {output_gif}...")
        frames = []
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            frame = env.unwrapped.render()
            frames.append(frame)

            action = get_action(model, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode Finished. Score: {total_reward}")
        print(f"Saving GIF to {output_gif}...")
        imageio.mimsave(output_gif, frames, fps=30)
        print(f"GIF {output_gif} saved successfully.")

    env.close()

if __name__ == "__main__":
    main()
