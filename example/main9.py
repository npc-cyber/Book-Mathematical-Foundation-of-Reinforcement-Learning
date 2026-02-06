import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 环境保持不变
class GridWorldPG:
    def __init__(self):
        self.height, self.width = 5, 5
        self.target = 24
        self.forbidden = 12
        self.actions = [0, 1, 2, 3, 4]

    def step(self, state, action):
        r, c = divmod(state, 5)
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1), (0,0)][action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5): return state, -1, False
        ns = nr * 5 + nc
        if ns == self.forbidden: return ns, -1, False
        if ns == self.target: return ns, 1, True
        return ns, -0.1, False # 给每一步一点小惩罚，鼓励快点走

# 策略网络：直接输出动作概率
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(25, 5) # 25个格子的One-hot输入，5个动作输出

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

def train_reinforce():
    env = GridWorldPG()
    net = PolicyNet()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    for episode in range(2000):
        state = 0 # 从起点开始
        log_probs = []
        rewards = []
        done = False
        
        # 1. 生成一个轨迹
        for _ in range(50): # 限制步数
            state_tensor = torch.zeros(25)
            state_tensor[state] = 1.0
            
            probs = net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample() # 这里就是你之前问的采样！
            
            next_state, reward, done = env.step(state, action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            if done: break
            
        # 2. 计算回报并更新 (REINFORCE)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + 0.9 * G
            returns.insert(0, G) # 前插入，保持顺序
        
        loss = 0
        for lp, g in zip(log_probs, returns):
            loss += -lp * g # 回报越高，对应的log_prob负值越小，即概率越大
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1} 完成训练")

    return net

def print_pg_policy(net):
    symbols = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
    print("\n[ 第9章 策略梯度 学习后的结果 ]:")
    for r in range(5):
        row = "|"
        for c in range(5):
            s = r * 5 + c
            state_tensor = torch.zeros(25)
            state_tensor[s] = 1.0
            with torch.no_grad():
                probs = net(state_tensor)
            a = torch.argmax(probs).item()
            if s == 24: row += " Goal |"
            elif s == 12: row += " [X]  |"
            else: row += f"{symbols[a]}|"
        print("-" * len(row)); print(row)
    print("-" * len(row))

if __name__ == "__main__":
    net = train_reinforce()
    print_pg_policy(net)