import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import json
import os

from agents.DCAE import DCAE
from utils import set_seed, EarlyStopping

def loss_empahsis_red(pred, truth):
    assert len(pred.shape) == 4
    expansion = 2
    diff = pred - truth
    diff[:, 0, :, :] *= expansion
    return torch.mean(diff**2)

if __name__ == '__main__':
    set_seed(0)
    
    pwd = os.path.dirname(os.path.abspath(__file__)) + "/"
    
    with open(pwd+"params.json", "r") as f:
        data = json.load(f)
    nsteps = data["nsteps"]
    hidden_dim = data["hidden_dim"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    img_size = (img_height, img_width, 3)
    
    env = gym.make("FetchReachDense-v2", render_mode="rgb_array", max_episode_steps=nsteps)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    agent = DCAE(img_size, hidden_dim).to(device)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((100, 100)),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    path = pwd + "model/DCAE_best.pth"
    
    batch_size = 128
    memory_size = 10000
    nepochs = 100
    
    seed = 42
    imgs = []

    print('Start Data collection')
    obs, info = env.reset(seed=seed)
    img = env.render()
    for i in range(memory_size):
        imgs.append(trans(img))
        action = env.action_space.sample()
        next_obs, reward, success, done, info = env.step(action)
        next_img = env.render()
        if success or done:
            obs, info = env.reset()
            img = env.render()
        else:
            obs = next_obs
            img = next_img
        if (i+1) % (int(memory_size/10)) == 0:
            print('%d / %d Data collected' % (i+1, memory_size))

    train_imgs, test_imgs = torch.utils.data.random_split(imgs, [0.7, 0.3])
    train_data = torch.utils.data.DataLoader(dataset=train_imgs, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(dataset=test_imgs, batch_size=batch_size, shuffle=False)
    # loss_function = loss_empahsis_red
    loss_function = F.mse_loss
    
    for epoch in range(nepochs):
        train_loss, test_loss = [], []
        for x in train_data:
            x = x.to(device)
            _, y = agent.forward(x, return_pred=True)
            loss = loss_function(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.cpu().detach().numpy())
        
        agent.eval()
        with torch.no_grad():
            for x in test_data:
                x = x.to(device)
                _, y = agent.forward(x, return_pred=True)
                loss = loss_function(y, x)
                test_loss.append(loss.cpu().detach().numpy())
        if (epoch+1) % (int(nepochs/10)) == 0:
            print("Epoch {}: Train Loss {}, Test Loss {}".format(epoch+1, np.mean(train_loss), np.mean(test_loss)))

    obs, info = env.reset()
    img = env.render()
    _, pred_img = agent.forward(trans(img).to(device), True)
    img = trans(img).numpy().transpose((1, 2, 0)) * 0.5 + 0.5
    pred_img = pred_img.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imshow(img)
    ax2 = fig.add_subplot(2, 1, 2)
    plt.imshow(pred_img)
    plt.show()

    env.close()