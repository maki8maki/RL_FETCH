import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
import torch
from torchvision import transforms

from agents.comb import DCAE_DDPG
from utils import set_seed

if __name__ == '__main__':
    set_seed(0)
    
    gamma = 0.7
    batch_size = 5
    memory_size = 100
    nepisodes = 10
    nsteps = 100
    hidden_dim = 20
    img_width = 80
    img_height = 80
    img_size = (img_height, img_width, 3)
    
    env = gym.make("FetchReachDense-v2", render_mode="rgb_array")
    
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
    
    frames = []
    
    print('Start Data collection')
    obs, info = env.reset(seed=42)
    img = env.render()
    frames.append(img)
    for _ in range(memory_size):
        state = {'image': trans(img).numpy(), 'obs': obs["observation"]}
        action = env.action_space.sample()
        next_obs, reward, success, done, info = env.step(action)
        next_img = env.render()
        frames.append(img)
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
    print('%d Data collected' % (memory_size))
    
    # plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    # patch = plt.imshow(frames[0])
    # plt.axis('off')

    # def animate(i):
    #     patch.set_data(frames[i])

    # anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    # plt.show()
    
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
        if episode % 5 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('DDPG: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()