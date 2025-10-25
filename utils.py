import torch
from torch import nn
import numpy as np
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def save_checkpoint(model, optimizer, epoch, avg_reward, path="flappy_bird_checkpoints"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"best_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_reward': avg_reward
    }, file_path)
    print(f"Checkpoint saved at {file_path}")

def log_performance(epoch, avg_reward, loss, path="flappy_bird/performance_log.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["epoch", "avg_reward", "loss"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([epoch, avg_reward, loss])

def smooth(data, window=5):
    return data.rolling(window, min_periods=1).mean()   

def plot_rewards(log_dir="flappy_bird/logs", window=5, save_path="flappy_bird/logs/reward_plot.png"):
    train_path = os.path.join(log_dir, "train_performance_log.csv")
    # infer_path = os.path.join(log_dir, "infer_performance_log.csv")

    plt.figure(figsize=(8, 5))

    if os.path.exists(train_path):
        try:
            df = pd.read_csv(train_path)
            if 'epoch' in df.columns and 'avg_reward' in df.columns:
                plt.plot(df["epoch"], smooth(df["avg_reward"], window), label="Train avg reward")
            else:
                print(f"[WARNING] Missing columns in {train_path}. Found: {list(df.columns)}")
        except Exception as e:
            print(f"[ERROR] Failed to read {train_path}: {e}")

    # if os.path.exists(infer_path):
    #     try:
    #         df = pd.read_csv(infer_path)
    #         if 'epoch' in df.columns and 'avg_reward' in df.columns:
    #             plt.plot(df["epoch"], smooth(df["avg_reward"], window), label="Inference avg reward")
    #         else:
    #             print(f"[WARNING] Missing columns in {infer_path}. Found: {list(df.columns)}")
    #     except Exception as e:
    #         print(f"[ERROR] Failed to read {infer_path}: {e}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Reward plot saved to {save_path}")

if __name__ == "__main__":
    plot_rewards("flappy_bird_ppo",10,"flappy_bird_ppo/logs/reward_plot.png")