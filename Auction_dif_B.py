import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

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
            # if w['b_i'] <= (B * 0.5) * (w['marginal'] / Tot_m):
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
        # print("计算第"+str(index)+"个人的报酬")

        remaining_workers = [copy.deepcopy(w_o) for w_o in original_workers if w_o['id'] != w_i['id']]
        new_selected_workers = select_workers(remaining_workers, B, m)
        # print("New Selected!")

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

            b_i_l_prime = min((w_l['b_i'] * w_i['marginal']) / w_l['marginal'], (B * w_i['marginal']) / (2 * prev_eq))
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments

def generate_workers(n, m):
    # 工人数据初始化
    workers = []
    for i in range(n):

        q_i = 1
        epsilon_i = np.random.choice(np.arange(1, 10, 1))  # 隐私预算
        # R_i = np.random.choice(np.arange(1, 10, 1))
        R_i = 1
        tasks = random.sample(range(m), random.choice([1, 2, 3]))  # 随机选k个任务
        # tasks = random.sample(range(m), random.choice([1]))  # 随机选k个任务

        # unit_task_cost = random.uniform(0.1, 1)
        unit_task_cost = 1
        unit_privacy_cost = 0.5

        b_i = len(tasks) * unit_task_cost + epsilon_i * unit_privacy_cost
        # 计算单位任务投标值
        unit_bid = b_i / len(tasks)
        if unit_bid >= (B_min / m):
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


def run_experiments(workers, B):

    workers_ori = copy.deepcopy(workers)
    selected_workers = select_workers(workers, B, m)
    payments = compute_payment(selected_workers, workers_ori, B, m)

    sum_mu = sum(worker['mu_i'] for worker in selected_workers)
    sum_lambda = sum(worker['lambda_i2'] for worker in selected_workers)
    tot_alpha = sum_mu + (sum_lambda / (1 - beta)) ** 0.5
    tot_pb = np.mean([worker['epsilon_i'] for worker in selected_workers])
    total_pay = sum(payments.values())

    return tot_alpha, tot_pb, total_pay


# 参数设置
n = 200  # 工人数量
m = 10  # 任务数量
Q_j = 3  # 每个任务的质量需求
B_values = [80, 90, 100, 110, 120, 130, 140, 150, 160] # 任务预算
B_min = np.min(B_values)
beta = 0.95 # 置信水平

num = 10
data_B = {B: [] for B in B_values}

# 执行num次实验
for experiment in range(num):
    print(f"执行第 {experiment + 1}/{num} 次实验...")
    # 生成一组新的工人
    workers = generate_workers(n, m)
    # 在相同的工人条件下，针对每个B运行实验
    for B in B_values:
        alpha, avg_pb, payment = run_experiments(copy.deepcopy(workers), B)
        data_B[B].append((alpha, avg_pb, payment))

all_data = []
for B, results in data_B.items():
    for i, (alpha, avg_pb, payment) in enumerate(results):
        all_data.append({
            "Experiment": i + 1,
            "B": B,
            "Alpha": alpha,
            "Avg_PB": avg_pb,
            "Payment": payment
        })

df = pd.DataFrame(all_data)
excel_filename = "Auction_dif_B_R1_all.xlsx"
df.to_excel(excel_filename, index=False)

# 计算平均值
final_results = []
for B in B_values:
    avg_alpha = np.mean([res[0] for res in data_B[B]])
    avg_pb = np.mean([res[1] for res in data_B[B]])
    avg_payment = np.mean([res[2] for res in data_B[B]])
    final_results.append((B, avg_alpha, avg_pb, avg_payment))

# 将结果保存到文件
df = pd.DataFrame(final_results, columns=['B', 'alpha', 'avg_pb', 'payment'])
df.to_excel("Auction_dif_B_R1.xlsx", index=False)