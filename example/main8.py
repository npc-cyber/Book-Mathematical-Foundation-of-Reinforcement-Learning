import numpy as np
import random

class GridWorldAppx:
    def __init__(self):
        self.height, self.width = 5, 5
        self.target = (4, 4)
        self.forbidden = (2, 2)
        self.actions = [0, 1, 2, 3, 4] # 上, 右, 下, 左, 原地

    def get_features(self, state, action):
        """将(s, a)转化为特征向量。这里使用简单的多项式特征或One-hot动作"""
        r, c = divmod(state, self.width)
        # 特征：[1, 行坐标, 列坐标, 动作One-hot]
        feat = np.zeros(8)
        feat[0] = 1.0
        feat[1] = r / 5.0
        feat[2] = c / 5.0
        feat[3 + action] = 1.0
        return feat

    def step(self, state, action):
        r, c = divmod(state, self.width)
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1), (0,0)][action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5): return state, -1, False
        ns = nr * 5 + nc
        if ns == 12: return ns, -1, False
        if ns == 24: return ns, 1, True
        return ns, 0, False

class AppxQLearner:
    def __init__(self, env, alpha=0.01, gamma=0.9, epsilon=0.2):
        self.env = env
        self.w = np.zeros(8) # 只有8个参数，而不是125个格
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        """计算 Q = w · x """
        feat = self.env.get_features(state, action)
        return np.dot(self.w, feat)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        qs = [self.get_q(state, a) for a in self.env.actions]
        return np.argmax(qs)

    def train(self, episodes=5000):
        for _ in range(episodes):
            s = random.randint(0, 23)
            done = False
            while not done:
                a = self.choose_action(s)
                next_s, r, done = self.env.step(s, a)
                
                # 计算 TD Target
                if done:
                    target = r
                else:
                    next_qs = [self.get_q(next_s, na) for na in self.env.actions]
                    target = r + self.gamma * max(next_qs)
                
                # 梯度下降更新参数 w
                # 线性函数的梯度就是特征向量本身 grad = x
                feat = self.env.get_features(s, a)
                self.w += self.alpha * (target - self.get_q(s, a)) * feat
                s = next_s
            if _ % 1000 == 0:
                print(f"Episode {_} completed.")

    def print_policy(self):
        arrows = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
        print("\n[ 函数近似(线性) 学习后的策略 ]:")
        for r in range(5):
            row = "|"
            for c in range(5):
                s = r * 5 + c
                if s == 24: row += " Goal |"
                elif s == 12: row += " [X]  |"
                else:
                    qs = [self.get_q(s, a) for a in self.env.actions]
                    row += f"{arrows[np.argmax(qs)]}|"
            print("-" * len(row)); print(row)
        print("-" * len(row))

if __name__ == "__main__":
    env = GridWorldAppx()
    solver = AppxQLearner(env)
    solver.train(10000)
    solver.print_policy()