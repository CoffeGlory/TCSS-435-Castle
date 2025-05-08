import gymnasium as gym
import torch
import numpy as np
import os
from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger


def main():
    # Hyperparameters
    num_episodes = 500
    env_name = "LunarLander-v3"
    gamma = 0.99 # Discount factor or return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent
    agent = DQNAgent(state_dim, action_dim, device)

    # Logging
    rewards_per_episode = []
    logger = TrainLogger()

    os.makedirs("results", exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_rewards = [] # To comput discounted return

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            episode_rewards.append(reward)

        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
        
        agent.decay_epsilon()
        success = 1 if total_reward >= 200 else 0
        
        if success: print(f"Episode {episode}: Successful landing!")

        logger.record(total_reward, G, success)

        rewards_per_episode.append(total_reward)

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Discounted Return: {G:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

    print("Saving model and rewards to 'results/'...")

    # Save results
    np.save("results/double_dqn_rewards.npy", np.array(rewards_per_episode))
    logger.save_to_csv("results/train_log_double_dqn.csv")
    torch.save(agent.q_network.state_dict(), "results/double_dqn_lunarlander.pth")  # ✅ Save model


def run_trained_model(agent_path: str = "results/double_dqn_lunarlander.pth", episodes: int = 5, render: bool = True):
    env_name = "LunarLander-v3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load trained agent
    agent = DQNAgent(state_dim, action_dim, device)
    agent.q_network.load_state_dict(torch.load(agent_path))
    agent.q_network.eval()
    agent.epsilon = 0.0

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        print(f"[TEST] Episode {episode}: Total Reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # train:
     main()

    # Run the last saved model
    # run_trained_model(agent_path="results/double_dqn_lunarlander.pth", episodes=3, render=True)
