import numpy as np
import matplotlib.pyplot as plt
import copy

class GridWorldBookModel:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.n_states = self.height * self.width
        self.n_actions = 5  # 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
        
        # 定义特殊区域
        self.target_state = 24  # (4,4) -> index 24
        self.forbidden_states = [12]  # (2,2) -> index 12 (书中的中间禁区)
        
        # 奖励设置 (参考书第3、4章常见设置)
        self.r_boundary = -1
        self.r_forbidden = -1
        self.r_target = 1
        self.r_step = 0
        
        # 动作效果 (dr, dc)
        self.action_effects = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)
        }

    def _get_pos(self, state_idx):
        return state_idx // self.width, state_idx % self.width

    def _get_idx(self, r, c):
        return r * self.width + c

    # 获取环境模型：P(s'|s,a) 和 R(s,a)
    # 返回一个字典: model[state][action] = (next_state, reward)
    # 注：书中的例子是确定性的，所以概率为1
    def get_model(self):
        model = {}
        for s in range(self.n_states):
            model[s] = {}
            for a in range(self.n_actions):
                r, c = self._get_pos(s)
                dr, dc = self.action_effects[a]
                next_r, next_c = r + dr, c + dc
                
                reward = self.r_step
                next_s = s 
                
                # 1. 检查边界
                if not (0 <= next_r < self.height and 0 <= next_c < self.width):
                    reward = self.r_boundary
                    next_s = s # 撞墙保持原地
                else:
                    next_s = self._get_idx(next_r, next_c)
                    
                    # 2. 检查禁区 (进入禁区给予惩罚，但允许进入)
                    if next_s in self.forbidden_states:
                        reward = self.r_forbidden
                    
                    # 3. 检查终点
                    if next_s == self.target_state:
                        reward = self.r_target
                        
                # 4. 特殊处理：如果在终点，任何动作都保持在终点且获得0奖励（吸收态）
                # 或者按照书中通常做法：到达终点后游戏结束，或者视为不断获得0
                # 这里为了Value Iteration计算，我们假设终点是吸收态，奖励为0
                if s == self.target_state:
                    next_s = s
                    reward = 0
                    
                model[s][a] = (next_s, reward)
        return model

def print_results(env, V, policy, title):
    print(f"\n{'='*40}")
    print(f"算法: {title}")
    print(f"{'='*40}")
    
    # 1. 打印价值表格
    print("\n[状态价值表 (State Values)]:")
    for r in range(env.height):
        row_str = "|"
        for c in range(env.width):
            s = r * env.width + c
            # 格式化输出：保留2位小数
            val_str = f"{V[s]:6.2f}"
            if s == env.target_state: val_str = "  Goal"
            if s in env.forbidden_states: val_str = f"{V[s]:6.2f}" # 禁区也有价值
            row_str += f" {val_str} |"
        print("-" * (len(row_str)-1))
        print(row_str)
    print("-" * (len(row_str)-1))

    # 2. 打印策略地图
    # 符号映射: 0:^, 1:>, 2:v, 3:<, 4:o
    arrows = {0: ' ↑ ', 1: ' → ', 2: ' ↓ ', 3: ' ← ', 4: ' o '}
    
    print("\n[最优策略图 (Optimal Policy)]:")
    for r in range(env.height):
        row_str = "|"
        for c in range(env.width):
            s = r * env.width + c
            if s == env.target_state:
                action_str = " OK " # 终点
            elif s in env.forbidden_states:
                # 禁区里也有策略，但也标记一下
                a = policy[s]
                action_str = f"[{arrows[a].strip()}]" 
            else:
                a = policy[s]
                action_str = arrows[a]
            row_str += f" {action_str} |"
        print("-" * (len(row_str)-1))
        print(row_str)
    print("-" * (len(row_str)-1))


def value_iteration(env : GridWorldBookModel, gamma=0.9, theta=1e-4):
    print("=== 开始价值迭代 (Value Iteration) ===")
    model = env.get_model()
    V = np.zeros(env.n_states)
    
    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for s in range(env.n_states):
            if s == env.target_state: continue # 终点价值通常设为0或不更新
            
            q_values = []
            for a in range(env.n_actions):
                next_s, r = model[s][a]
                # Bellman Optimality Equation
                q = r + gamma * V[next_s]
                q_values.append(q)
            
            new_V[s] = max(q_values)
            delta = max(delta, abs(new_V[s] - V[s]))
            
        V = new_V
        iteration += 1
        if delta < theta:
            break
            
    print(f"价值迭代收敛于第 {iteration} 轮")
    
    # 提取最优策略
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q_values = []
        for a in range(env.n_actions):
            next_s, r = model[s][a]
            q = r + gamma * V[next_s]
            q_values.append(q)
        policy[s] = np.argmax(q_values)
        
    return V, policy

def policy_iteration(env, gamma=0.9, theta=1e-4):
    print("=== 开始策略迭代 (Policy Iteration) ===")
    model = env.get_model()
    V = np.zeros(env.n_states)
    # 初始化一个随机策略 (每个状态默认动作0)
    policy = np.zeros(env.n_states, dtype=int)
    
    iteration = 0
    while True:
        # 1. 策略评估 (Policy Evaluation)
        # 一直迭代直到 V 收敛到当前策略的真实 V_pi
        eval_iter = 0
        while True:
            delta = 0
            for s in range(env.n_states):
                if s == env.target_state: continue
                
                a = policy[s] # 当前策略选择的动作
                next_s, r = model[s][a]
                
                v_new = r + gamma * V[next_s]
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            eval_iter += 1
            if delta < theta:
                break
        
        # 2. 策略改进 (Policy Improvement)
        policy_stable = True
        for s in range(env.n_states):
            if s == env.target_state: continue
            
            old_action = policy[s]
            
            # 贪婪选择最好的动作
            q_values = []
            for a in range(env.n_actions):
                next_s, r = model[s][a]
                q = r + gamma * V[next_s]
                q_values.append(q)
            
            best_action = np.argmax(q_values)
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        iteration += 1
        if policy_stable:
            break
            
    print(f"策略迭代收敛于第 {iteration} 轮 (外层循环)")
    return V, policy

def truncated_policy_iteration(env, max_eval_steps=5, gamma=0.9, theta=1e-4):
    print(f"=== 开始截断策略迭代 (Truncated PI, k={max_eval_steps}) ===")
    model = env.get_model()
    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)
    
    iteration = 0
    # 设置一个最大迭代次数防止死循环（虽然一般会收敛）
    for i in range(1000): 
        # 1. 截断策略评估 (Truncated Policy Evaluation)
        # 只跑 max_eval_steps 步，不求完全收敛
        for _ in range(max_eval_steps):
            for s in range(env.n_states):
                if s == env.target_state: continue
                a = policy[s]
                next_s, r = model[s][a]
                V[s] = r + gamma * V[next_s]
        
        # 2. 策略改进 (Policy Improvement)
        policy_stable = True
        # 同时我们要检查 Value 是否收敛，作为整个算法的停止条件
        # 在截断策略迭代中，通常判定 V 的变化很小即可
        v_delta = 0
        
        for s in range(env.n_states):
            if s == env.target_state: continue
            
            old_action = policy[s]
            old_v = V[s]
            
            # 计算所有动作的 Q 值
            q_values = []
            for a in range(env.n_actions):
                next_s, r = model[s][a]
                q = r + gamma * V[next_s]
                q_values.append(q)
            
            best_action = np.argmax(q_values)
            policy[s] = best_action
            
            # 更新 V 为最佳动作的价值 (这一步让它更像 Value Iteration)
            # 或者，标准的截断PI可能在这里不更新V，只更新Policy。
            # 但根据书中的算法 4.3 (Truncated policy iteration algorithm)：
            # Value update 其实是在评估阶段做的。
            # 策略改进阶段只更新策略。
            
            # 为了计算收敛，我们看看 best Q 和当前 V 的差距
            best_val = max(q_values)
            v_delta = max(v_delta, abs(best_val - V[s]))

        iteration += 1
        if v_delta < theta:
            break
            
    print(f"截断策略迭代收敛于第 {iteration} 轮")
    return V, policy

if __name__ == "__main__":
    env = GridWorldBookModel()
    
    # 1. 运行价值迭代
    v_vi, p_vi = value_iteration(env)
    print_results(env, v_vi, p_vi, title="Value Iteration")
    print("-" * 30)
    
    # 2. 运行策略迭代
    v_pi, p_pi = policy_iteration(env)
    print_results(env, v_pi, p_pi, title="Policy Iteration")
    print("-" * 30)
    
    # 3. 运行截断策略迭代 (评估步数设为 3)
    v_tpi, p_tpi = truncated_policy_iteration(env, max_eval_steps=3)
    print_results(env, v_tpi, p_tpi, title="Truncated Policy Iteration (3 steps)")