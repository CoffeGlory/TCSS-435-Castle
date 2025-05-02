import csv
from typing import List, Tuple

class TrainLogger:
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []
        self.success_flags: List[int] = []

    def record(self, reward: float, ret: float, success: int):
        """Record metrics from a single episode."""
        self.episode_rewards.append(reward)
        self.episode_returns.append(ret)
        self.success_flags.append(success)

    def get_metrics(self) -> Tuple[List[float], List[float], List[int]]:
        """Get all stored metrics."""
        return self.episode_rewards, self.episode_returns, self.success_flags

    def save_to_csv(self, filename: str = "results/train_log.csv"):
        """Save all recorded data to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode_Reward", "Episode_Return", "Success"])
            for r, g, s in zip(self.episode_rewards, self.episode_returns, self.success_flags):
                writer.writerow([r, g, s])
