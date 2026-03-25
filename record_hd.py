import torch
import torch.nn as nn
import os
import gym
import procgen
import numpy as np
from torch.distributions.categorical import Categorical

# 1. Path Configuration
RUN_ID = "starpilot__ppo_procgen__1__1774388309"
CHECKPOINT_PATH = f"runs/{RUN_ID}/ppo_procgen.cleanrl_model" 
OUTPUT_FOLDER = "portfolio_hd_videos"

# 2. Impala Architecture (Matches your 10M step brain)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_shape[0], out_channels=out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        h, w, c = 64, 64, 3
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seqs.append(ConvSequence(shape, out_channels))
            shape = (out_channels, (shape[1] + 1) // 2, (shape[2] + 1) // 2)
        self.network = nn.Sequential(*conv_seqs, nn.ReLU(), nn.Flatten(), nn.Linear(out_channels * shape[1] * shape[2], 256), nn.ReLU())
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# 3. Execution
# ... (Keep the Agent and Impala classes from the previous script) ...

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try levels 1 through 20 to find the best gameplay
    for level_seed in range(1, 21):
        env = gym.make("procgen-starpilot-v0", render_mode="rgb_array", start_level=level_seed)
        # Only record if the agent actually survives and scores
        env = gym.wrappers.RecordVideo(env, f"{OUTPUT_FOLDER}/seed_{level_seed}", episode_trigger=lambda x: True)
        
        agent = Agent(env.action_space.n).to(device)
        agent.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        agent.eval()

        obs = env.reset()
        done = False
        score = 0
        
        print(f"--- Testing Seed {level_seed} ---")
        while not done:
            with torch.no_grad():
                # Logic: Convert HWC to CHW and normalize
                action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).permute(2, 0, 1).unsqueeze(0).to(device))
            obs, reward, done, _ = env.step(action.cpu().numpy()[0])
            score += reward
            
        print(f"Seed {level_seed} Finished with Score: {score}")
        env.close()
        
        # If the agent did well, we stop here. We found our "Portfolio" video!
        if score > 15:
            print(f"WINNER: Seed {level_seed} is a great run. Video saved.")
            break