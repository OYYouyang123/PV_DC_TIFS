import numpy as np
import random
import copy
import pandas as pd

# ===================== 差分隐私位置扰动：指数机制 =====================
def MeanVar(epsilon_i, R_i, delta_r=0.01, delta_theta=np.pi / 180, num_samples=1000):
    true_x, true_y = 0, 0
    r_values = np.arange(delta_r, R_i + delta_r, delta_r)
    theta_values = np.arange(0, 2 * np.pi, delta_theta)

    candidate_points = [(true_x + r * np.cos(theta), true_y + r * np.sin(theta), r)
                        for r in r_values for theta in theta_values]

    distances = np.array([np.sqrt((x - true_x) ** 2 + (y - true_y) ** 2)
                          for x, y, _ in candidate_points])
    scores = np.exp(epsilon_i / R_i * (R_i - distances))
    probabilities = scores / np.sum(scores)

    sampled_idx = np.random.choice(len(candidate_points), size=num_samples, p=probabilities)
    mapped_positions = [candidate_points[i][:2] for i in sampled_idx]
    sampled_distances = [np.sqrt((x - true_x) ** 2 + (y - true_y) ** 2) for x, y in mapped_positions]

    return np.mean(sampled_distances), np.var(sampled_distances)


# ===================== 工人选择算法 =====================
def select_workers(c_workers, B, m, Q_j):
    global global_record_times

    task_remaining = {j: Q_j for j in range(m)}
    s_workers = []
    prev_eq = 0

    while not all(task_remaining[j] <= 0 for j in range(m)):
        if not c_workers:
            global_record_times += 1
            break

        # 计算贡献值
        for w in c_workers:
            q_Add = sum(min(w['q_i'], task_remaining[t]) for t in w['tasks'] if task_remaining[t] > 0)
            w['marginal'] = q_Add / w['w_loss']
            w['fitness'] = (w['marginal'] / w['b_i']) if w['b_i'] > 0 else -np.inf

        # 选择 fitness 最大者
        c_workers.sort(key=lambda x: x['fitness'], reverse=True)

        picked = False
        for index, w in enumerate(c_workers):
            Tot_m = prev_eq + w['marginal']
            if Tot_m <= 0:
                continue

            if w['b_i'] <= max((B * 0.5) * (w['marginal'] / Tot_m), B * 0.05):
                best_worker = c_workers.pop(index)
                s_workers.append(best_worker)
                prev_eq += best_worker['marginal']

                # 更新任务剩余需求
                for ti in best_worker['tasks']:
                    if task_remaining[ti] > 0:
                        contribute = min(best_worker['q_i'], task_remaining[ti])
                        task_remaining[ti] -= contribute

                picked = True
                break

        if not picked:
            break

    return s_workers


# ===================== 工人支付计算 =====================
def compute_payment(s_workers, original_workers, B, m, Q_j):
    payments = {}

    for w_i in s_workers:
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

            denom = (w_l['marginal'] + 1e-12)
            denom2 = (2 * prev_eq + 1e-12)
            b_i_l_prime = min((w_l['b_i'] * w_i['marginal']) / denom,
                              (B * w_i['marginal']) / denom2)
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments


# ===================== 参数设置 =====================
n = 150
m = 12
B = 300
beta = 0.95
num_rounds = 301
num_executions = 10

global_record_times = 0

# 输入数据（只读一次即可）
excel_file = "output_data_transposed_TEMP_selected_rows.xlsx"
data_from_excel = pd.read_excel(excel_file, sheet_name=None)

# TEMP 物理合理范围
TEMP_MIN, TEMP_MAX = 220.0, 320.0


# # ===================== 变量组合循环 =====================
# Qj_list = [3, 4, 5]
# normal_ratios = [0.6, 0.7, 0.8, 0.9]
Qj_list = [3, 4, 5]
normal_ratios = [0.6]

for Q_j in Qj_list:
    for normal_ratio in normal_ratios:
        print(f"\n===================== Q_j = {Q_j}, normal_ratio = {normal_ratio} =====================")

        final_mae = np.zeros(num_rounds)
        all_mae_per_execution = []  # list of list，每个元素长度=num_rounds

        # ===================== 多次实验 =====================
        for execution in range(num_executions):
            print(f"\n--- (Q_j={Q_j}, normal_ratio={normal_ratio}) 执行第 {execution + 1} 次实验 ---")

            unit_tb = B / m
            workers = []

            # 工人类型（由 normal_ratio 决定）
            num_normal = int(normal_ratio * n)
            num_malicious = n - num_normal
            worker_types = np.array(['normal'] * num_normal + ['malicious'] * num_malicious)
            np.random.shuffle(worker_types)

            # 不同类型噪声方差分布
            lambda_nor_temp = 1.0 / 1
            lambda_mal_temp = 1.0 / 9
            scales = np.where(worker_types == 'normal', 1 / lambda_nor_temp, 1 / lambda_mal_temp)
            worker_sigma2 = np.random.exponential(scale=scales)

            # 初始化工人
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

            mae_per_round = []

            # ===================== 主循环 =====================
            for t in range(num_rounds):
                print(f"\n正在进行第 {t} 轮循环...")

                workers_all = copy.deepcopy(workers)

                # 任务真值
                true_task_values = []
                for _, df in data_from_excel.items():
                    true_task_values.append(df.iloc[t, 0])
                true_task_values = np.array(true_task_values, dtype=float)

                # 工人任务选择 + 投标过滤
                for i in range(n):
                    tasks = random.sample(range(m), random.choice([1, 2, 3]))
                    unit_task_cost = 1
                    unit_privacy_cost = 0.5
                    b_i = len(tasks) * unit_task_cost + workers_all[i]['epsilon_i'] * unit_privacy_cost
                    unit_bid = b_i / len(tasks)

                    if unit_bid >= unit_tb:
                        continue

                    workers_all[i]['tasks'] = tasks
                    workers_all[i]['b_i'] = b_i

                # 选择与支付
                workers_ori = copy.deepcopy(workers_all)
                selected_workers = select_workers(workers_all, B, m, Q_j)
                print(f"选中 {len(selected_workers)} 名工人")

                payments = compute_payment(selected_workers, workers_ori, B, m, Q_j)
                total_payment = sum(payments.values())
                print(f"总支付报酬: {total_payment:.2f}")


                # ===================== 上报数据：高斯噪声 + TEMP裁剪 =====================
                submitted_data = []
                submitted_task_ids = []

                for worker in selected_workers:
                    sigma2 = worker['sigma2']
                    for task_id in worker['tasks']:
                        tv = float(true_task_values[task_id])

                        noise = np.random.normal(0.0, np.sqrt(sigma2))
                        reported = tv + noise

                        # 物理合理范围裁剪（温度K）
                        reported = float(np.clip(reported, TEMP_MIN, TEMP_MAX))

                        submitted_data.append(reported)
                        submitted_task_ids.append(int(task_id))


                submitted_data = np.array(submitted_data, dtype=float)
                submitted_task_ids = np.array(submitted_task_ids, dtype=int)

                # ===================== 对比方案：均值聚合 =====================
                estimated_task_values = np.full(m, np.nan, dtype=float)
                for j in range(m):
                    idx = np.where(submitted_task_ids == j)[0]
                    if len(idx) > 0:
                        estimated_task_values[j] = np.mean(submitted_data[idx])

                mae = np.nanmean(np.abs(estimated_task_values - true_task_values))
                mae_per_round.append(mae)

            final_mae += np.array(mae_per_round, dtype=float)
            all_mae_per_execution.append(list(mae_per_round))

        # ===================== 保存结果 =====================
        final_mae /= num_executions

        ratio_str = f"{normal_ratio:.1f}"
        prefix = f"TE(Temp)-baseline-mean-Q{Q_j}-nor{ratio_str}"

        # 1) 存多次实验平均后的曲线
        with pd.ExcelWriter(f"{prefix}.xlsx") as writer:
            pd.DataFrame({"Round": range(num_rounds), "Average MAE": final_mae}).to_excel(
                writer, sheet_name="MAE", index=False
            )

        all_mae_df = pd.DataFrame(
            {f"Exec_{k+1}": all_mae_per_execution[k] for k in range(len(all_mae_per_execution))}
        )
        all_mae_df.insert(0, "Round", range(num_rounds))
        all_mae_df.to_excel(f"{prefix}-all.xlsx", index=False)