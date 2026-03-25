import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import os
from pathlib import Path

# --- WINDOWS PATHING ---
PROJECT_ROOT = Path(__file__).parent
VIDEO_FOLDER = PROJECT_ROOT / "videos"
# Update this to your actual .pth file path (e.g., "runs/c51/model.pth")
MODEL_PATH = PROJECT_ROOT / "runs" / "your_folder" / "model.pth" 

def record_agent():
    # 1. Ensure the video folder exists
    if not VIDEO_FOLDER.exists():
        os.makedirs(VIDEO_FOLDER)

    # 2. Setup Environment (rgb_array is required for recording)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # This wrapper films the agent
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_FOLDER), 
        name_prefix="industrial_demo",
        episode_trigger=lambda x: True # Record every episode
    )

    print(f"📦 Looking for model at: {MODEL_PATH}")
    
    # 3. Running the simulation
    obs, info = env.reset()
    done = False
    
    print("🎥 Recording... The window might stay hidden, that's normal.")

    while not done:
        # REPLACE with: action = agent.predict(obs) 
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # 4. CRITICAL: env.close() must be called to finalize the MP4 file
    env.close()
    print(f"✅ Video successfully saved to: {VIDEO_FOLDER}")

if __name__ == "__main__":
    record_agent()