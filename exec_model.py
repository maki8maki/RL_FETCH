import gymnasium as gym
import torch
from torchvision import transforms
import json
import os

from agents.comb import DCAE_DDPG
from utils import set_seed, anim

if __name__ == '__main__':
    set_seed(0)
    
    pwd = os.path.dirname(os.path.abspath(__file__)) + "/"
    
    with open(pwd+"params.json", "r") as f:
        data = json.load(f)
    gamma = data["gamma"]
    batch_size = data["batch_size"]
    memory_size = data["memory_size"]
    nepisodes = data["nepisodes"]
    nsteps = data["nsteps"]
    hidden_dim = data["hidden_dim"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    img_size = (img_height, img_width, 3)
    
    env = gym.make("FetchReachDense-v2", render_mode="rgb_array", max_episode_steps=nsteps)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    agent = DCAE_DDPG(img_size, hidden_dim, env.observation_space["observation"],
                      env.action_space, gamma=gamma, batch_size=batch_size,
                      memory_size=memory_size, device=device)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    path = pwd + "model/DCAE_DDPG_best.pth"
    
    frames = []
    
    agent.load(path)
    agent.eval()
    
    for i in range(5):
        obs, info = env.reset()
        img = env.render()
        while True:
            frames.append(img)
            state = {'image': trans(img).numpy(), 'obs': obs["observation"]}
            action = agent.get_action(state).cpu().detach().numpy()
            next_obs, reward, success, done, info = env.step(action)
            next_img = env.render()
            if success or done:
                break
            else:
                img = next_img
                obs = next_obs
    
    anim(frames)

    env.close()