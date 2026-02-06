import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 环境设置 (5x5 Grid World)
class GridWorldAC:
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
        return ns, -0.01, False 

# 定义 Actor-Critic 网络
class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.common = nn.Linear(25, 64)   # 公共特征层
        self.actor = nn.Linear(64, 5)     # Actor Head: 输出概率
        self.critic = nn.Linear(64, 1)    # Critic Head: 输出 V 值

    def forward(self, x):
        x = F.relu(self.common(x))
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

def train_a2c():
    env = GridWorldAC()
    model = ActorCriticNet()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    gamma = 0.9

    for episode in range(2000):
        state = 0
        done = False
        
        while not done:
            state_tensor = torch.zeros(25); state_tensor[state] = 1.0
            
            # 1. 获取动作概率和当前 V 值
            probs, value = model(state_tensor)
            
            # 2. 采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # 3. 与环境交互
            next_state, reward, done = env.step(state, action.item())
            
            # 4. 获取下一个状态的 V 值
            next_state_tensor = torch.zeros(25); next_state_tensor[next_state] = 1.0
            _, next_value = model(next_state_tensor)
            
            # 5. 计算 TD Error (Advantage)
            # 如果结束了，TD目标就是 reward，否则是 r + gamma * V(s')
            target = reward + (gamma * next_value.item() if not done else 0)
            delta = target - value.item()
            
            # 6. 计算损失函数
            # Actor Loss: -log_prob * delta
            # Critic Loss: MSE(value, target)
            actor_loss = -dist.log_prob(action) * delta
            target_tensor = torch.tensor([target], dtype=torch.float32)
            critic_loss = F.mse_loss(value, target_tensor)
            loss = actor_loss + critic_loss
            
            # 7. 更新模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1} 训练中...")

    return model

def print_ac_policy(model):
    symbols = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
    print("\n[ 第10章 Actor-Critic 学习后的策略 ]:")
    for r in range(5):
        row = "|"
        for c in range(5):
            s = r * 5 + c
            state_tensor = torch.zeros(25); state_tensor[s] = 1.0
            with torch.no_grad():
                probs, _ = model(state_tensor)
            a = torch.argmax(probs).item()
            if s == 24: row += " Goal |"
            elif s == 12: row += " [X]  |"
            else: row += f"{symbols[a]}|"
        print("-" * len(row)); print(row)
    print("-" * len(row))

if __name__ == "__main__":
    trained_model = train_a2c()
    print_ac_policy(trained_model)