import numpy as np
import random

# 环境逻辑 (5x5 网格)
class GridWorldTD:
    def __init__(self):
        self.height, self.width = 5, 5
        self.target = (4, 4)
        self.forbidden = (2, 2)
        self.actions = [0, 1, 2, 3, 4] # 上, 右, 下, 左, 原地

    def step(self, state, action):
        r, c = divmod(state, self.width)
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1), (0,0)][action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5):
            return state, -1, False
        ns = nr * 5 + nc
        if ns == self.forbidden[0]*5 + self.forbidden[1]:
            return ns, -1, False
        if ns == self.target[0]*5 + self.target[1]:
            return ns, 1, True
        return ns, 0, False

# TD 学习器
class TDSolver:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((25, 5))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        return np.argmax(self.Q[state])

    # --- 算法 1: Sarsa ---
    def train_sarsa(self, episodes=5000):
        for _ in range(episodes):
            s = random.randint(0, 23)
            a = self.choose_action(s)
            done = False
            while not done:
                next_s, r, done = self.env.step(s, a)
                next_a = self.choose_action(next_s)
                # Sarsa 更新核心：使用实际选取的下个动作 next_a
                td_error = r + self.gamma * self.Q[next_s, next_a] - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error
                s, a = next_s, next_a

    # --- 算法 2: Q-learning ---
    def train_q_learning(self, episodes=5000):
        for _ in range(episodes):
            s = random.randint(0, 23)
            done = False
            while not done:
                a = self.choose_action(s)
                next_s, r, done = self.env.step(s, a)
                # Q-learning 更新核心：使用最优的 max Q
                td_error = r + self.gamma * np.max(self.Q[next_s]) - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error
                s = next_s

    def print_policy(self, name):
        symbols = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
        print(f"\n[{name} 学习后的策略]:")
        for r in range(5):
            row = "|"
            for c in range(5):
                s = r * 5 + c
                if s == 24: row += " Goal |"
                else: row += f"{symbols[np.argmax(self.Q[s])]}|"
            print("-" * len(row))
            print(row)

# 运行
if __name__ == "__main__":
    env = GridWorldTD()
    
    # Q-learning 实验
    q_solver = TDSolver(env)
    q_solver.train_q_learning(5000)
    q_solver.print_policy("Q-learning")
    
    # Sarsa 实验
    s_solver = TDSolver(env)
    s_solver.train_sarsa(5000)
    s_solver.print_policy("Sarsa")