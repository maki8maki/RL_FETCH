import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from torchvision import transforms
import json
import os

from agents.comb import DCAE_DDPG
from utils import set_seed, anim, EarlyStopping

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
    early_stopping = EarlyStopping(verbose=True, patience=100, path=path)
    
    frames = []
    seed = 42
    
    print('Start Data collection')
    obs, info = env.reset(seed=seed)
    img = env.render()
    for i in range(memory_size):
        # frames.append(img)
        state = {'image': trans(img).numpy(), 'obs': obs["observation"]}
        action = env.action_space.sample()
        next_obs, reward, success, done, info = env.step(action)
        next_img = env.render()
        next_state = {'image': trans(next_img).numpy(), 'obs': next_obs["observation"]}
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'success': int(success),
            'done': int(done)
        }
        agent.ddpg.replay_buffer.append(transition)
        if success or done:
            obs, info = env.reset()
            img = env.render()
        else:
            obs = next_obs
            img = next_img
        if (i+1) % (int(memory_size/10)) == 0:
            print('%d / %d Data collected' % (i+1, memory_size))
    
    # anim(frames)
    
    episode_rewards = []
    num_average_epidodes = 5
    for episode in range(nepisodes):
        obs, info = env.reset()
        img = env.render()
        episode_reward = 0
        for t in range(nsteps):
            state = {"image": trans(img).numpy(), "obs": obs["observation"]}
            action = agent.get_action(state).cpu().detach().numpy()
            next_obs, reward, success, done, info = env.step(action)
            next_img = env.render()
            next_state = {"image": trans(next_img).numpy(), "obs": next_obs["observation"]}
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'success': int(success),
                'done': int(done)
            }
            agent.ddpg.replay_buffer.append(transition)
            episode_reward += reward
            agent.update()
            if success or done:
                break
            else:
                obs = next_obs
                img = next_img
        episode_rewards.append(episode_reward)
        if (episode+1) % 5 == 0:
            print("Episode %d finished | Episode reward %f" % (episode+1, episode_reward))
        early_stopping(-episode_reward, agent)
        if early_stopping.early_stop:
            break

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('DDPG: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()
    
    frames = []
    
    agent.load(path)
    agent.eval()
    obs, info = env.reset(seed=seed)
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