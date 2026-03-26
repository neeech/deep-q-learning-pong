import argparse
import os
import gymnasium as gym
import ale_py
import torch
import numpy as np
import time

from dql import DQN

def get_action(model, obs, device) -> int:
    """Choose the best action from the model."""
    with torch.no_grad():
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t)
        return int(q_values.argmax(dim=1).item())


def main():
    parser = argparse.ArgumentParser(description='Play Pong with a trained DQN agent.')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Filename of the checkpoint in the checkpoints directory')
    args = parser.parse_args()

    checkpoint_path = os.path.join("checkpoints", args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading weights from: {checkpoint_path}")
    
    # Environment setup
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84)
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    n_actions = env.action_space.n
    model = DQN(n_actions).to(device)
    
    try:
        # Load the weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # The error indicates the weights are under the 'model_state_dict' key
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval() # Set to evaluation mode
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        env.close()
        return

    print("Starting game... Close the window to stop.")
    
    try:
        while True: # Play indefinitely
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = get_action(model, obs, device)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            print(f"Episode Finished. Score: {total_reward}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nGame window closed or error occurred: {e}")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
