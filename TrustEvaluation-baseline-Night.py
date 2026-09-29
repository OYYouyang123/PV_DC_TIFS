import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt


# **基于指数机制的位置扰动（差分隐私）**
# 功能：使用指数机制模拟差分隐私位置扰动，计算扰动后距离的均值和方差。
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
# 功能：在预算 B 下选择一组能最大程度满足任务质量需求的工人。
def select_workers(c_workers, B, m):
    global global_record_times
    global Q_j

    task_remaining = {j: Q_j for j in range(m)}
    s_workers = []
    prev_eq = 0

    while not all(task_remaining[j] <= 0 for j in range(m)):
        if not c_workers:
            global_record_times = global_record_times + 1
            break

        # 计算贡献值
        for w in c_workers:
            q_Add = sum(min(w['q_i'], task_remaining[t]) for t in w['tasks'] if task_remaining[t] > 0)
            w['marginal'] = q_Add / w['w_loss']  # 计算工人贡献（有效质量）
            w['fitness'] = w['marginal'] / w['b_i']

        # 选择贡献值/投标值最大者
        c_workers.sort(key=lambda x: x['fitness'], reverse=True)

        for index, w in enumerate(c_workers):
            Tot_m = prev_eq + w['marginal']
            if w['b_i'] <= max((B * 0.5) * (w['marginal'] / (Tot_m + 1e-12)), B * 0.05):
                best_worker = c_workers.pop(index)  # 选择最佳工人
                s_workers.append(best_worker)
                prev_eq += best_worker['marginal']

                # 更新任务剩余质量需求
                for ti in best_worker['tasks']:
                    if task_remaining[ti] > 0:
                        contribute = min(best_worker['q_i'], task_remaining[ti])
                        task_remaining[ti] -= contribute
                break
            else:
                # 可选：如果输出太多，可以注释掉
                print("预算不满足")

    return s_workers


# **工人支付报酬计算**
def compute_payment(s_workers, original_workers, B, m):
    global Q_j

    payments = {}  # 存储每个工人的支付金额

    for index, w_i in enumerate(s_workers):
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
                (w_l['b_i'] * w_i['marginal']) / (w_l['marginal'] + 1e-12),
                (B * w_i['marginal']) / (2 * prev_eq + 1e-12)
            )
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments


# =========================
# 固定参数（不随组合变化）
# =========================
n = 150  # 工人数量
m = 12   # 任务数量
B = 300  # 预算总量

beta = 0.95  # 置信水平
xi = 0.1  # 时间衰减系数
delta_q = 0.1  # 奖惩因子
epsilon = 1e-4  # 收敛阈值（TD内部收敛判断）
error_threshold = 3  # 信任评估中的误差阈值
max_inner_iter = 30  # TD内部最大迭代次数
num_rounds = 301  # 每次实验轮数
num_executions = 10  # 每个组合重复实验次数

# 数据文件
excel_file = "output_data_Route5_6.xlsx"
data_from_excel = pd.read_excel(excel_file, sheet_name=None)

# 组合参数
Qj_list = [3, 4, 5]
normal_ratio_list = [0.6, 0.7, 0.8, 0.9]

# （可选）为了不同组合可复现：给每个组合设置不同随机种子基准
base_seed = 2026

# 批量跑组合
for Qj_val in Qj_list:
    for normal_ratio in normal_ratio_list:
        # ========== 组合参数赋值 ==========
        Q_j = Qj_val  # 注意：select_workers/compute_payment 使用的是全局 Q_j

        global_record_times = 0

        # 每个组合单独的统计数组
        final_mae = np.zeros(num_rounds)  # 平均MAE
        final_quality_normal = np.zeros(num_rounds + 1)  # Normal工人平均质量
        final_quality_malicious = np.zeros(num_rounds + 1)  # Malicious工人平均质量

        print("\n" + "=" * 80)
        print(f"Running combo: Q_j={Q_j}, normal_ratio={normal_ratio}")
        print("=" * 80)

        # =========================
        # 重复执行（多次实验取均值）
        # =========================
        for execution in range(num_executions):
            # 可选：固定每次 execution 的随机性（便于对比不同组合）
            seed = base_seed + 1000 * Q_j + int(normal_ratio * 100) * 10 + execution
            np.random.seed(seed)
            random.seed(seed)

            print(f"\n📌--- Q_j={Q_j}, ratio={normal_ratio} | 执行第 {execution + 1}/{num_executions} 次实验 ---")

            # 初始化工人
            unit_tb = B / m
            workers = []

            # 设置工人类型：normal 和 malicious
            num_normal = int(normal_ratio * n)
            num_malicious = n - num_normal
            worker_types = np.array(['normal'] * num_normal + ['malicious'] * num_malicious)
            np.random.shuffle(worker_types)

            # 根据工人类型确定指数分布参数 ：λ_nor=1, λ_mal=0.05
            lambda_nor = 1.0
            lambda_mal = 0.05
            scales = np.where(worker_types == 'normal', 1 / lambda_nor, 1 / lambda_mal)
            worker_sigma2 = np.random.exponential(scale=scales)

            # 工人质量初始预估
            worker_estimated_quality = np.ones(n, dtype=float)
            quality_history = np.zeros((n, num_rounds + 1))
            quality_history[:, 0] = 1
            mae_per_round = []

            # 工人数据初始化
            for i in range(n):
                q_i = 1
                R_i = 5
                epsilon_i = np.random.choice(np.arange(1, 10, 1))
                mu_i, lambda_i2 = MeanVar(epsilon_i, R_i)
                workers.append({
                    'id': i,
                    'epsilon_i': epsilon_i,
                    'R_i': R_i,
                    'q_i': q_i,
                    'mu_i': mu_i,
                    'lambda_i2': lambda_i2,
                    'w_loss': mu_i + np.sqrt(lambda_i2) / ((1 - beta) ** 0.5),
                    'type': worker_types[i],
                    'sigma2': worker_sigma2[i],
                })

            # 主循环（共循环 num_rounds 次）
            for t in range(num_rounds):
                workers_all = copy.deepcopy(workers)
                print(f"\n📌 [Q_j={Q_j}, ratio={normal_ratio}] 正在进行第 {t} 轮循环...")

                # 感知任务真值（每个 sheet 对应一个任务）
                true_task_values = []
                for array_name, df in data_from_excel.items():
                    true_task_values.append(df.iloc[t, 0])

                candidates = []
                worker_tasks = []  # 可选保留

                for i in range(n):
                    tasks = random.sample(range(m), random.choice([1, 2, 3]))
                    worker_tasks.append(tasks)

                    unit_task_cost = 1
                    unit_privacy_cost = 0.5
                    b_i = len(tasks) * unit_task_cost + workers_all[i]['epsilon_i'] * unit_privacy_cost
                    unit_bid = b_i / len(tasks)

                    if unit_bid >= unit_tb:
                        continue

                    w = workers_all[i]
                    w['tasks'] = tasks
                    w['b_i'] = b_i
                    candidates.append(w)

                # 候选为空会导致后续无法提交数据，这里可直接记一次并跳过本轮
                if len(candidates) == 0:
                    global_record_times += 1
                    mae_per_round.append(np.nan)
                    quality_history[:, t + 1] = worker_estimated_quality
                    continue

                workers_ori = copy.deepcopy(candidates)
                selected_workers = select_workers(candidates, B, m)

                print(f"选中 {len(selected_workers)} 名工人")
                payments = compute_payment(selected_workers, workers_ori, B, m)
                total_payment = sum(payments.values())
                print(f"\n总支付报酬: {total_payment:.2f}")

                # 如果本轮没选到工人，同样无法提交数据：直接记录并进入下一轮
                if len(selected_workers) == 0:
                    mae_per_round.append(np.nan)
                    quality_history[:, t + 1] = worker_estimated_quality
                    continue

                submitted_data = []
                submitted_task_ids = []
                worker_indices = []
                selected_worker_ids = [w['id'] for w in selected_workers]

                for worker in selected_workers:
                    worker_id = worker['id']
                    sigma2 = worker['sigma2']
                    tasks = worker['tasks']

                    for task_id in tasks:
                        noise = np.random.normal(0, np.sqrt(sigma2))
                        data = int(round(true_task_values[task_id] + noise))
                        submitted_data.append(data)
                        submitted_task_ids.append(task_id)
                        worker_indices.append(worker_id)

                submitted_data = np.array(submitted_data)
                submitted_task_ids = np.array(submitted_task_ids)
                worker_indices = np.array(worker_indices)

                # Truth Discovery 阶段 (CRH方法)
                quality_for_td = worker_estimated_quality[worker_indices].copy()

                # 初始化任务估计值
                estimated_task_values = np.full(m, np.nan)
                for j in range(m):
                    idx = np.where(submitted_task_ids == j)[0]
                    if len(idx) > 0:
                        weights_j = quality_for_td[idx]
                        data_j = submitted_data[idx]
                        estimated_task_values[j] = np.sum(weights_j * data_j) / (np.sum(weights_j) + 1e-6)

                # TD 内循环
                for iter_td in range(max_inner_iter):
                    distance = np.abs(submitted_data - estimated_task_values[submitted_task_ids])
                    quality_for_td = np.exp(-(-np.log(0.5) / 5) * distance)
                    quality_for_td = (quality_for_td - quality_for_td.min()) / (
                        quality_for_td.max() - quality_for_td.min() + 1e-8
                    )

                    prev_estimated_task_values = estimated_task_values.copy()
                    for j in range(m):
                        idx = np.where(submitted_task_ids == j)[0]
                        if len(idx) > 0:
                            weights_j = quality_for_td[idx]
                            data_j = submitted_data[idx]
                            estimated_task_values[j] = np.sum(weights_j * data_j) / (np.sum(weights_j) + 1e-6)
                        else:
                            estimated_task_values[j] = np.nan

                    if np.linalg.norm(estimated_task_values - prev_estimated_task_values, ord=1) < epsilon:
                        break

                # 评估 MAE
                mae = np.nanmean(np.abs(estimated_task_values - true_task_values))
                mae_per_round.append(mae)

                # 计算误差/得分
                estimated_errors = np.abs(submitted_data - estimated_task_values[submitted_task_ids])
                z_scores = error_threshold - estimated_errors
                S_i_all = np.tanh(z_scores)

                S_i_sum = np.zeros(n)
                S_i_count = np.zeros(n)

                for idx, worker_id in enumerate(worker_indices):
                    S_i_sum[worker_id] += S_i_all[idx]
                    S_i_count[worker_id] += 1

                S_i_final = S_i_sum / (S_i_count + 1e-6)

                # 更新参与任务工人的质量
                worker_estimated_quality[selected_worker_ids] = (
                    np.exp(-xi) * worker_estimated_quality[selected_worker_ids]
                    + S_i_final[selected_worker_ids] * delta_q
                )

                # 归一化工人质量
                q_min = worker_estimated_quality.min()
                q_max = worker_estimated_quality.max()
                worker_estimated_quality = (worker_estimated_quality - q_min) / (q_max - q_min + 1e-6)

                # 保存历史记录
                quality_history[:, t + 1] = worker_estimated_quality

            # execution 聚合
            final_mae += np.array(mae_per_round, dtype=float)
            final_quality_normal += np.mean(quality_history[np.where(worker_types == 'normal')[0], :], axis=0)
            final_quality_malicious += np.mean(quality_history[np.where(worker_types == 'malicious')[0], :], axis=0)

        # =========================
        # 组合结果取平均并写入 Excel（文件名体现组合）
        # =========================
        final_mae /= num_executions
        final_quality_normal /= num_executions
        final_quality_malicious /= num_executions

        out_name = f"TE-baseline_Night_Qj{Q_j}_ratio{normal_ratio:.1f}.xlsx"
        with pd.ExcelWriter(out_name) as writer:
            pd.DataFrame({
                "Round": range(num_rounds),
                "Average MAE": final_mae
            }).to_excel(writer, sheet_name="MAE", index=False)

            pd.DataFrame({
                "Round": range(num_rounds + 1),
                "Normal Average Quality": final_quality_normal,
                "Malicious Average Quality": final_quality_malicious
            }).to_excel(writer, sheet_name="Average Quality", index=False)

            # 可选：把本组合的参数也存一张表，方便回溯
            pd.DataFrame([{
                "Q_j": Q_j,
                "normal_ratio": normal_ratio,
                "n": n,
                "m": m,
                "B": B,
                "num_rounds": num_rounds,
                "num_executions": num_executions,
                "global_record_times": global_record_times
            }]).to_excel(writer, sheet_name="Config", index=False)

        print(f"\n✅ Combo finished. Saved: {out_name}")