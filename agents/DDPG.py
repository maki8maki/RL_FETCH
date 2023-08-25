import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.utils import ReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16) -> None:
        super(ActorNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float)
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.shape[0])

    def forward(self, s):
        h = F.elu(self.fc1(s))
        h = F.elu(self.fc2(h))
        a = self.action_mean + self.action_halfwidth*torch.tanh(self.fc3(h))
        return a

class CriticNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16):
        super(CriticNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float)
        self.fc1 = nn.Linear(num_state+action_space.shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.shape[0])

    def forward(self, s, a):
        a = (a-self.action_mean)/self.action_halfwidth
        h = F.elu(self.fc1(torch.cat([s,a],1)))
        h = F.elu(self.fc2(h))
        q = self.fc3(h)
        return q

class DDPG:
    def __init__(self, observation_space, action_space, gamma=0.99, lr=1e-3, batch_size=32, memory_size=50000, device="cpu"):
        self.num_state = observation_space.shape[0]
        self.num_action = action_space.shape[0]
        # self.state_mean = 0.5*(observation_space.high + observation_space.low)
        # self.state_halfwidth = 0.5*(observation_space.high - observation_space.low)
        self.gamma = gamma  # 割引率
        self.batch_size = batch_size
        self.actor = ActorNetwork(self.num_state, action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)  # actorのターゲットネットワーク
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = CriticNetwork(self.num_state, action_space).to(device)
        self.critic_target = copy.deepcopy(self.critic)  # criticのターゲットネットワーク
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

    # 連続値の状態を[-1,1]の範囲に正規化
    def normalize_state(self, state):
        state = (state-self.state_mean)/self.state_halfwidth
        return state

    # リプレイバッファからサンプルされたミニバッチをtensorに変換
    def batch_to_tensor(self, batch):
        # states = torch.tensor([self.normalize_state(s) for s in batch["states"]], dtype=torch.float)
        states = torch.tensor(batch["states"], dtype=torch.float)
        actions = torch.tensor(batch["actions"], dtype=torch.float)
        # next_states = torch.tensor([self.normalize_state(s) for s in batch["next_states"]], dtype=torch.float)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float)
        return states, actions, next_states, rewards

    # actorとcriticを更新
    def update(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards = self.batch_to_tensor(batch)
        # criticの更新
        target_q = (rewards + self.gamma*self.critic_target(next_states, self.actor_target(next_states)).squeeze()).data
        q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # actorの更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
    
    def update_from_batch(self, batch):
        states, actions, next_states, rewards = self.batch_to_tensor(batch)
        # criticの更新
        target_q = (rewards + self.gamma*self.critic_target(next_states, self.actor_target(next_states)).squeeze()).data
        q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # actorの更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

    # Q値が最大の行動を選択
    def get_action(self, state):
        state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float).view(-1, self.num_state)
        action = self.actor(state_tensor).view(self.num_action)
        return action