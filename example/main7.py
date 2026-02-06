import numpy as np
import random

# 环境逻辑与之前保持一致
class GridWorldNStep:
    def __init__(self):
        self.height, self.width = 5, 5
        self.target = (4, 4)
        self.forbidden = (2, 2)
        self.actions = [0, 1, 2, 3, 4]

    def step(self, state, action):
        r, c = divmod(state, self.width)
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1), (0,0)][action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5): return state, -1, False
        ns = nr * 5 + nc
        if ns == self.forbidden[0]*5 + self.forbidden[1]: return ns, -1, False
        if ns == self.target[0]*5 + self.target[1]: return ns, 1, True
        return ns, 0, False

class NStepSarsa:
    def __init__(self, env, n=3, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.n = n # n 步
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((25, 5))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        return np.argmax(self.Q[state])

    def train(self, episodes=3000):
        for _ in range(episodes):
            state = random.randint(0, 23)
            action = self.choose_action(state)
            
            # 存储轨迹的列表: (s, a, r)
            states, actions, rewards = [state], [action], []
            
            T = float('inf')
            t = 0
            while True:
                if t < T:
                    next_state, reward, done = self.env.step(states[t], actions[t])
                    states.append(next_state)
                    rewards.append(reward)
                    
                    if done:
                        print(f"done at t={t}")
                        T = t + 1
                    else:
                        actions.append(self.choose_action(next_state))
                
                # tau 是我们要更新的那个时刻 (t - n + 1)
                tau = t - self.n + 1
                if tau >= 0:
                    # 计算 n 步回报 G
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i-1]
                    
                    # 如果还没到终点，加上最后一项预测值
                    if tau + self.n < T:
                        G += (self.gamma ** self.n) * self.Q[states[tau + self.n], actions[tau + self.n]]
                    
                    # 更新 Q 表
                    s_tau, a_tau = states[tau], actions[tau]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])
                
                if tau == T - 1: print(f"tau {tau} break"); break 
                t += 1

    def print_policy(self):
        arrows = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
        print(f"\n[ {self.n} 步 Sarsa 学习后的策略 ]:")
        for r in range(5):
            row = "|"
            for c in range(5):
                s = r * 5 + c
                if s == 24: row += " Goal |"
                elif s == 12: row += " [X]  |" # 标记禁区
                else: row += f"{arrows[np.argmax(self.Q[s])]}|"
            print("-" * len(row)); print(row)
        print("-" * len(row))

if __name__ == "__main__":
    env = GridWorldNStep()
    # 试试 3 步 Sarsa
    solver = NStepSarsa(env, n=3)
    solver.train(5000)
    solver.print_policy()