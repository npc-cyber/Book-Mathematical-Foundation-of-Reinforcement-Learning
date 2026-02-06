import numpy as np
import random

# ==========================================
# 1. 基础环境 (恢复到试错模式)
# ==========================================
class GridWorldMC:
    def __init__(self):
        self.height, self.width = 5, 5
        self.target = (4, 4)
        self.forbidden = (2, 2)
        self.actions = [0, 1, 2, 3, 4] # 上, 右, 下, 左, 原地

    def step(self, state, action):
        r, c = divmod(state, self.width)
        dr, dc = [(-1,0), (0,1), (1,0), (0,-1), (0,0)][action]
        nr, nc = r + dr, c + dc
        
        # 边界与禁区逻辑
        if not (0 <= nr < 5 and 0 <= nc < 5):
            return state, -1, False
        ns = nr * 5 + nc
        if ns == self.forbidden[0]*5 + self.forbidden[1]:
            return ns, -1, False
        if ns == self.target[0]*5 + self.target[1]:
            return ns, 1, True # 到达终点
        return ns, 0, False

# ==========================================
# 2. 算法实现
# ==========================================

class MonteCarloSolver:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.height * env.width, len(env.actions)))
        self.returns = {(s, a): [] for s in range(25) for a in env.actions}
        self.policy = np.random.randint(0, 5, 25)

    # 生成一个回合的数据 (Trajectory)
    def generate_episode(self, epsilon=0.1):
        trajectory = []
        state = random.randint(0, 23) # 随机起点 (除了终点)
        done = False
        steps = 0
        while not done and steps < 50:
            # epsilon-greedy 探索
            if random.random() < epsilon:
                action = random.choice(self.env.actions)
            else:
                action = self.policy[state]
            
            next_state, reward, done = self.env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            steps += 1
        return trajectory

    # 算法实现: MC Control (Every-visit)
    def train(self, num_episodes=5000, epsilon=0.1):
        for i in range(num_episodes):
            episode = self.generate_episode(epsilon)
            G = 0
            # 从后往前计算回报，这是 MC 的经典做法
            visited_sa = set()
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = r + self.gamma * G
                
                # 书中提到的 Every-visit 或 First-visit 逻辑
                # 这里记录 (s, a) 的回报均值
                self.returns[(s, a)].append(G)
                self.Q[s, a] = np.mean(self.returns[(s, a)])
                
                # 策略改进: 直接 greedy
                self.policy[s] = np.argmax(self.Q[s])
            
            if (i+1) % 1000 == 0:
                print(f"已完成 {i+1} 个回合的学习...")

    def print_policy(self):
        symbols = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
        print("\n[MC 学习后的策略地图]:")
        for r in range(5):
            row = "|"
            for c in range(5):
                s = r * 5 + c
                if s == 24: row += " Goal |"
                else: row += f"{symbols[self.policy[s]]}|"
            print("-" * len(row))
            print(row)
        print("-" * len(row))

# ==========================================
# 3. 运行
# ==========================================
if __name__ == "__main__":
    env = GridWorldMC()
    solver = MonteCarloSolver(env)
    
    # 训练模型
    solver.train(num_episodes=10000, epsilon=0.2)
    
    # 输出结果
    solver.print_policy()