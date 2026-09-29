import numpy as np
import random
import copy
import pandas as pd

# **基于指数机制的位置扰动**
def MeanVar(epsilon_i, R_i, delta_r=0.01, delta_theta=np.pi / 180, num_samples=1000):
    true_x, true_y = 0, 0
    r_values = np.arange(delta_r, R_i + delta_r, delta_r)
    theta_values = np.arange(0, 2 * np.pi, delta_theta)
    candidate_points = [(true_x + r * np.cos(theta), true_y + r * np.sin(theta), r)
                        for r in r_values for theta in theta_values]

    distances = np.array([np.sqrt((x - true_x) ** 2 + (y - true_y) ** 2) for x, y, _ in candidate_points])
    scores = np.exp(epsilon_i / R_i * (R_i - distances))
    probabilities = scores / np.sum(scores)

    sampled_points = np.random.choice(len(candidate_points), size=num_samples, p=probabilities)
    mapped_positions = [candidate_points[i][:2] for i in sampled_points]
    distances = [np.sqrt((x - true_x) ** 2 + (y - true_y) ** 2) for x, y in mapped_positions]

    return np.mean(distances), np.var(distances)


# **工人选择算法**
def select_workers(workers, B, m, Q_j):

    task_remaining = {j: Q_j for j in range(m)}  # 记录任务剩余质量需求
    selected_workers = []  # 记录最终选中的工人
    prev_eq = 0  # 记录累计的所有工人边缘贡献值

    while not all(task_remaining[j] <= 0 for j in range(m)):

        if not workers:
            break

        # 计算贡献值
        for w in workers:
            q_Add = sum(min(w['q_i'], task_remaining[t]) for t in w['tasks'] if task_remaining[t] > 0)
            w['marginal'] = q_Add / w['w_loss']  # 计算工人贡献（有效质量）
            w['fitness'] = w['marginal'] / w['b_i']

        # 选择贡献值/投标值最大者
        workers.sort(key=lambda x: x['fitness'], reverse=True)

        picked = False
        for index, w in enumerate(workers):
            Tot_m = prev_eq + w['marginal']
            if w['b_i'] <= max((B * 0.5) * (w['marginal'] / Tot_m), B * 0.05):
                best_worker = workers.pop(index)  # 选择最佳工人
                selected_workers.append(best_worker)
                prev_eq += best_worker['marginal']
                # 更新任务剩余质量需求
                for ti in best_worker['tasks']:
                    if task_remaining[ti] > 0:
                        contribute = min(best_worker['q_i'], task_remaining[ti])
                        task_remaining[ti] -= contribute
                picked = True
                break

        if not picked:
            # 所有人都不满足预算约束则终止，避免死循环
            break

    return selected_workers


def compute_payment(selected_workers, original_workers, B, m, Q_j):

    payments = {}  # 存储每个工人的支付金额

    for w_i in selected_workers:
        remaining_workers = [copy.deepcopy(w_o) for w_o in original_workers if w_o['id'] != w_i['id']]
        new_selected_workers = select_workers(remaining_workers, B, m, Q_j)

        task_remaining = {j: Q_j for j in range(m)}
        prev_eq = 0
        max_bid_i = 0

        for w_l in new_selected_workers:
            q_Add_l = sum(min(w_l['q_i'], task_remaining[tl]) for tl in w_l['tasks'] if task_remaining[tl] > 0)
            w_l['marginal'] = q_Add_l / w_l['w_loss']

            q_Add_i = sum(min(w_i['q_i'], task_remaining[ti]) for ti in w_i['tasks'] if task_remaining[ti] > 0)
            w_i['marginal'] = q_Add_i / w_i['w_loss']

            for tl in w_l['tasks']:
                if task_remaining[tl] > 0:
                    contribute = min(w_l['q_i'], task_remaining[tl])
                    task_remaining[tl] -= contribute

            prev_eq += w_l['marginal']

            # 防止除零（理论上 w_l['marginal'] 不应为0，但加一层保险）
            if w_l['marginal'] <= 0:
                continue

            b_i_l_prime = min(
                (w_l['b_i'] * w_i['marginal']) / w_l['marginal'],
                (B * w_i['marginal']) / (2 * prev_eq) if prev_eq > 0 else 0
            )
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments


# =========================
# 参数设置（除beta外固定）
# =========================
n = 200          # 工人数量
Q_j = 3          # 每个任务的质量需求
B = 160          # 预算总量
Pt_list = [0.6, 0.7, 0.8]
Ps1, Ps2, Ps3 = 0.3, 0.5, 0.7
repeat_times = 10

# 固定 m = 10
m = 10
unit_tb = B / m

# 让 beta 变化
beta_list = [0.5, 0.6, 0.7, 0.8, 0.95]

# === 主实验循环（beta维度）===
all_results = []

for beta in beta_list:
    print(f"\n📌 正在进行 beta = {beta} 的实验...")

    for Pt in Pt_list:
        print(f"  ⏳ 正在执行 Pt = {Pt} 的 {repeat_times} 次实验...")

        # 累计结果
        stat = {
            '全部验证': {'loss': 0, 'privacy': 0, 'k': 0},
            '未验证': {'loss': 0, 'privacy': 0, 'k': 0},
            '部分验证1': {'loss': 0, 'privacy': 0, 'k': 0},
            '部分验证2': {'loss': 0, 'privacy': 0, 'k': 0},
            '部分验证3': {'loss': 0, 'privacy': 0, 'k': 0},
        }

        for r_t in range(repeat_times):
            workers = []

            for i in range(n):
                q_i = 1
                epsilon_i = np.random.choice(np.arange(1, 10, 1))
                R_i = 5
                tasks = random.sample(range(m), random.choice([1, 2, 3]))

                unit_task_cost = 1
                unit_privacy_cost = 0.5
                b_i = len(tasks) * unit_task_cost + epsilon_i * unit_privacy_cost

                unit_bid = b_i / len(tasks)
                if unit_bid >= unit_tb:
                    continue

                mu_i, lambda_i2 = MeanVar(epsilon_i, R_i)
                # beta 在这里影响 w_loss（风险项）
                w_loss = mu_i + np.sqrt(lambda_i2) / ((1 - beta) ** 0.5)

                workers.append({
                    'id': i,
                    'epsilon_i': epsilon_i,
                    'R_i': R_i,
                    'tasks': tasks,
                    'q_i': q_i,
                    'b_i': b_i,
                    'mu_i': mu_i,
                    'lambda_i2': lambda_i2,
                    'w_loss': w_loss,
                })

            workers_ori = copy.deepcopy(workers)
            selected_workers = select_workers(workers, B, m, Q_j)
            payments = compute_payment(selected_workers, workers_ori, B, m, Q_j)
            _total_payment = sum(payments.values())  # 如需也记录可加入结果表

            # 诚实/作弊 epsilon
            for worker in selected_workers:
                honest = np.random.rand() < Pt
                worker['real_epsilon_i'] = worker['epsilon_i'] if honest else np.random.randint(1, worker['epsilon_i'] + 1)
                worker['honest'] = honest

            k = len(selected_workers)

            # 对比不同验证方案下的总地理损失和平均隐私水平
            # 采用全部验证方案
            sum_mu = sum(w['mu_i'] for w in selected_workers)
            sum_lambda = sum(w['lambda_i2'] for w in selected_workers)
            tot_alpha = sum_mu + np.sqrt(sum_lambda / (1 - beta))
            avg_privacy = np.mean([w['epsilon_i'] for w in selected_workers]) if k > 0 else np.nan
            stat['全部验证']['loss'] += tot_alpha
            stat['全部验证']['privacy'] += (0 if np.isnan(avg_privacy) else avg_privacy)
            stat['全部验证']['k'] += k

            # 采用不验证方案
            sum_mu = 0
            sum_lambda = 0
            real_epsilons = []
            for w in selected_workers:
                mu, lambd = MeanVar(w['real_epsilon_i'], w['R_i'])
                sum_mu += mu
                sum_lambda += lambd
                real_epsilons.append(w['real_epsilon_i'])

            tot_alpha = sum_mu + np.sqrt(sum_lambda / (1 - beta))
            avg_privacy = np.mean(real_epsilons) if k > 0 else np.nan
            stat['未验证']['loss'] += tot_alpha
            stat['未验证']['privacy'] += (0 if np.isnan(avg_privacy) else avg_privacy)
            stat['未验证']['k'] += k

            # 采用部分验证方案1
            sum_mu = 0
            sum_lambda = 0
            partial_epsilons = []
            for w in selected_workers:
                epsilon_used = w['epsilon_i'] if random.random() < Ps1 else w['real_epsilon_i']
                mu, lambd = MeanVar(epsilon_used, w['R_i'])
                sum_mu += mu
                sum_lambda += lambd
                partial_epsilons.append(epsilon_used)
            tot_alpha = sum_mu + np.sqrt(sum_lambda / (1 - beta))
            avg_privacy = np.mean(partial_epsilons) if k > 0 else np.nan
            stat['部分验证1']['loss'] += tot_alpha
            stat['部分验证1']['privacy'] += (0 if np.isnan(avg_privacy) else avg_privacy)
            stat['部分验证1']['k'] += k

            # 采用部分验证方案2
            sum_mu = 0
            sum_lambda = 0
            partial_epsilons = []
            for w in selected_workers:
                epsilon_used = w['epsilon_i'] if random.random() < Ps2 else w['real_epsilon_i']
                mu, lambd = MeanVar(epsilon_used, w['R_i'])
                sum_mu += mu
                sum_lambda += lambd
                partial_epsilons.append(epsilon_used)
            tot_alpha = sum_mu + np.sqrt(sum_lambda / (1 - beta))
            avg_privacy = np.mean(partial_epsilons) if k > 0 else np.nan
            stat['部分验证2']['loss'] += tot_alpha
            stat['部分验证2']['privacy'] += (0 if np.isnan(avg_privacy) else avg_privacy)
            stat['部分验证2']['k'] += k

            # 采用部分验证方案3
            sum_mu = 0
            sum_lambda = 0
            partial_epsilons = []
            for w in selected_workers:
                epsilon_used = w['epsilon_i'] if random.random() < Ps3 else w['real_epsilon_i']
                mu, lambd = MeanVar(epsilon_used, w['R_i'])
                sum_mu += mu
                sum_lambda += lambd
                partial_epsilons.append(epsilon_used)
            tot_alpha = sum_mu + np.sqrt(sum_lambda / (1 - beta))
            avg_privacy = np.mean(partial_epsilons) if k > 0 else np.nan
            stat['部分验证3']['loss'] += tot_alpha
            stat['部分验证3']['privacy'] += (0 if np.isnan(avg_privacy) else avg_privacy)
            stat['部分验证3']['k'] += k

        # 平均结果记录
        for scheme in ['全部验证', '未验证', '部分验证1', '部分验证2', '部分验证3']:
            all_results.append({
                'm': m,
                'beta': beta,
                'Pt': Pt,
                '方案': scheme,
                '总地理损失': stat[scheme]['loss'] / repeat_times,
                '平均隐私水平': stat[scheme]['privacy'] / repeat_times,
                '选中工人数(平均)': stat[scheme]['k'] / repeat_times
            })

# === 保存到 Excel ===
df = pd.DataFrame(all_results)
out_name = "Verification_difbeta.xlsx"
df.to_excel(out_name, index=False)
print(f"\n✅ 所有实验完成，结果已保存为 {out_name}")