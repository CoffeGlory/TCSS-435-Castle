import gymnasium as gym

env = gym.make('LunarLander-v3')
obs, info = env.reset()
print(obs)

for _ in range(10):
    action = env.action_space.sample()  # random action
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        break
env.close()
