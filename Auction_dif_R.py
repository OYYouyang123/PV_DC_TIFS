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
def select_workers(workers, B, m):

    task_remaining = {j: Q_j for j in range(m)}  # 记录任务剩余质量需求
    selected_workers = []  # 记录最终选中的工人
    prev_eq = 0  # 记录累计的所有工人边缘贡献值

    while not all(task_remaining[j] <= 0 for j in range(m)):  # 确保每个任务都满足质量需求

        if not workers:  # 若没有工人可选，则跳出
            break

        # 计算贡献值
        for w in workers:
            q_Add = sum(min(w['q_i'], task_remaining[t]) for t in w['tasks'] if task_remaining[t] > 0)
            w['marginal'] = q_Add / w['w_loss']  # 计算工人贡献（有效质量）
            w['fitness'] = w['marginal'] / w['b_i']

        # 选择贡献值/投标值最大者
        workers.sort(key=lambda x: x['fitness'], reverse=True)

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
                break
            else:
                print("预算不满足")

    return selected_workers


def compute_payment(selected_workers, original_workers, B, m):

    payments = {}  # 存储每个工人的支付金额

    for index, w_i in enumerate(selected_workers):

        remaining_workers = [copy.deepcopy(w_o) for w_o in original_workers if w_o['id'] != w_i['id']]
        new_selected_workers = select_workers(remaining_workers, B, m)

        task_remaining = {j: Q_j for j in range(m)}
        prev_eq = 0
        max_bid_i = 0

        for w_l in new_selected_workers:
            q_Add = sum(min(w_l['q_i'], task_remaining[tl]) for tl in w_l['tasks'] if task_remaining[tl] > 0)
            w_l['marginal'] = q_Add / w_l['w_loss']

            q_Add = sum(min(w_i['q_i'], task_remaining[ti]) for ti in w_i['tasks'] if task_remaining[ti] > 0)
            w_i['marginal'] = q_Add / w_i['w_loss']

            for tl in w_l['tasks']:
                if task_remaining[tl] > 0:
                    contribute = min(w_l['q_i'], task_remaining[tl])
                    task_remaining[tl] -= contribute

            prev_eq += w_l['marginal']

            b_i_l_prime = min(
                (w_l['b_i'] * w_i['marginal']) / w_l['marginal'],
                (B * w_i['marginal']) / (2 * prev_eq)
            )
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments


def generate_workers(n, m, R_fixed):
    # 工人数据初始化
    workers = []
    for i in range(n):

        q_i = 1
        epsilon_i = np.random.choice(np.arange(1, 10, 1))  # 隐私预算

        # 关键修改：R_i 不再写死为 1，而是使用外部传入的固定值
        R_i = R_fixed

        tasks = random.sample(range(m), random.choice([1, 2, 3]))  # 随机选k个任务

        unit_task_cost = random.uniform(0.1, 1)
        unit_privacy_cost = 0.5

        b_i = len(tasks) * unit_task_cost + epsilon_i * unit_privacy_cost

        # 计算单位任务投标值
        unit_bid = b_i / len(tasks)
        if unit_bid >= (B / (m * Q_j)):
            continue  # 跳过该工人

        mu_i, lambda_i2 = MeanVar(epsilon_i, R_i)
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
            'w_loss': w_loss
        })

    return workers


def run_experiments(n, R_fixed, num=10):

    alpha_results = []
    pb_results = []
    pay_results = []
    experiment_details = []

    for i in range(num):
        print(f"n={n}, R_i={R_fixed}, 运行第 {i+1}/{num} 次实验...")

        workers = generate_workers(n, m, R_fixed)
        workers_ori = copy.deepcopy(workers)
        selected_workers = select_workers(workers, B, m)

        tot_pb = np.mean([worker['epsilon_i'] for worker in selected_workers]) if selected_workers else 0.0

        payments = compute_payment(selected_workers, workers_ori, B, m) if selected_workers else {}
        total_pay = sum(payments.values())

        sum_mu = sum(worker['mu_i'] for worker in selected_workers)
        sum_lambda = sum(worker['lambda_i2'] for worker in selected_workers)
        tot_alpha = sum_mu + (sum_lambda / (1 - beta)) ** 0.5 if selected_workers else 0.0

        alpha_results.append(tot_alpha)
        pb_results.append(tot_pb)
        pay_results.append(total_pay)

        experiment_details.append({
            "Experiment": i + 1,
            "n": n,
            "R_i": R_fixed,
            "tot_alpha": tot_alpha,
            "tot_pb": tot_pb,
            "total_pay": total_pay,
            "num_selected": len(selected_workers)
        })

    alpha = float(np.mean(alpha_results)) if alpha_results else 0.0
    avg_pb = float(np.mean(pb_results)) if pb_results else 0.0
    payment = float(np.mean(pay_results)) if pay_results else 0.0

    return n, R_fixed, alpha, avg_pb, payment, experiment_details


# 参数设置（保持不变）
n = 200 # 工人数量
m = 10   # 任务数量
Q_j = 3  # 每个任务的质量需求
B = 160  # 预算总量
beta = 0.95  # 隐私水平


R_values = list(range(1, 11, 1))  # 1,2,...,10

data_R = []
all_experiment_results = []

for R in R_values:
    result = run_experiments(n, R_fixed=R, num=10)
    # result: (n, R_i, alpha, avg_pb, payment, details)
    summary_row, details = result[:5], result[5]
    data_R.append(summary_row)
    all_experiment_results.extend(details)

# 保存汇总结果（不同 R_i 的平均指标）
df = pd.DataFrame(data_R, columns=['n', 'R_i', 'alpha', 'avg_pb', 'payment'])
df.to_excel("Auction_test_dif_R.xlsx", index=False)

# 保存每次实验详细结果
df_details = pd.DataFrame(all_experiment_results)
df_details.to_excel("Auction_test_dif_R_all.xlsx", index=False)