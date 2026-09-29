import numpy as np
import random
import copy
import pandas as pd

# =========================
# 差分隐私位置扰动：指数机制
# =========================
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


# =========================
# 工人选择算法
# =========================
def select_workers(c_workers, B, m, Q_j):
    global global_record_times

    task_remaining = {j: Q_j for j in range(m)}
    s_workers = []
    prev_eq = 0

    while not all(task_remaining[j] <= 0 for j in range(m)):
        if not c_workers:
            global_record_times += 1
            break

        for w in c_workers:
            q_Add = sum(min(w['q_i'], task_remaining[t]) for t in w['tasks'] if task_remaining[t] > 0)
            w['marginal'] = q_Add / w['w_loss']
            w['fitness'] = (w['marginal'] / w['b_i']) if w['b_i'] > 0 else -np.inf

        c_workers.sort(key=lambda x: x['fitness'], reverse=True)

        picked = False
        for index, w in enumerate(c_workers):
            Tot_m = prev_eq + w['marginal']
            if Tot_m <= 0:
                continue

            if w['b_i'] <= max((B * 0.5) * (w['marginal'] / (Tot_m + 1e-12)), B * 0.05):
                best_worker = c_workers.pop(index)
                s_workers.append(best_worker)
                prev_eq += best_worker['marginal']

                for ti in best_worker['tasks']:
                    if task_remaining[ti] > 0:
                        contribute = min(best_worker['q_i'], task_remaining[ti])
                        task_remaining[ti] -= contribute

                picked = True
                break

        if not picked:
            break

    return s_workers


# =========================
# 工人支付计算
# =========================
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
            b_i_l_prime = min(
                (w_l['b_i'] * w_i['marginal']) / denom,
                (B * w_i['marginal']) / denom2
            )
            max_bid_i = max(max_bid_i, b_i_l_prime)

        payments[w_i['id']] = max_bid_i

    return payments


# =========================
# 参数设置（固定）
# =========================
n = 150
m = 12
B = 300

beta = 0.95
xi = 0.1
delta_q = 0.1
epsilon_td = 1e-4
error_threshold = 4
max_inner_iter = 30

num_rounds = 301
num_executions = 10

global_record_times = 0

# =========================
# AQI 数据集读取
# =========================
excel_file = "output_data_AQI3_m12_r301_skipNaN.xlsx"
data_from_excel = pd.read_excel(excel_file, sheet_name=None)

sheet_names = sorted(data_from_excel.keys())
if len(sheet_names) != m:
    raise ValueError(f"任务数不一致：Excel里sheet数={len(sheet_names)}，代码设定 m={m}")

AQI_MIN, AQI_MAX = 0.0, 500.0

# =========================
# 组合参数
# =========================
Qj_list = [3, 4, 5]
normal_ratio_list = [0.6]
q_thre_list = [0.3, 0.4, 0.5, 0.6]

base_seed = 2026

for Q_j in Qj_list:
    for normal_ratio in normal_ratio_list:
        for q_thre in q_thre_list:
            global_record_times = 0

            final_mae = np.zeros(num_rounds, dtype=float)
            final_quality_normal = np.zeros(num_rounds + 1, dtype=float)
            final_quality_malicious = np.zeros(num_rounds + 1, dtype=float)

            print("\n" + "=" * 80)
            print(f"Running combo (AQI-PV-DC): Q_j={Q_j}, normal_ratio={normal_ratio}, q_thre={q_thre}")
            print("=" * 80)

            for execution in range(num_executions):
                seed = (base_seed
                        + 1000 * Q_j
                        + int(normal_ratio * 100) * 10
                        + int(q_thre * 100) * 100
                        + execution)
                np.random.seed(seed)
                random.seed(seed)

                unit_tb = B / m
                workers = []

                # 工人类型
                num_normal = int(normal_ratio * n)
                num_malicious = n - num_normal
                worker_types = np.array(['normal'] * num_normal + ['malicious'] * num_malicious)
                np.random.shuffle(worker_types)

                # AQI场景噪声方差（与均值baseline一致）
                lambda_nor_aqi = 1.0 / 2.0
                lambda_mal_aqi = 1.0 / 36.0
                scales = np.where(worker_types == 'normal', 1 / lambda_nor_aqi, 1 / lambda_mal_aqi)
                worker_sigma2 = np.random.exponential(scale=scales)

                # 质量初始化
                worker_estimated_quality = np.ones(n, dtype=float)
                quality_history = np.zeros((n, num_rounds + 1), dtype=float)
                quality_history[:, 0] = 1.0

                mae_per_round = []

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

                for t in range(num_rounds):
                    workers_all = copy.deepcopy(workers)

                    # 任务真值（固定sheet顺序）
                    true_task_values = np.array(
                        [data_from_excel[name].iloc[t, 0] for name in sheet_names],
                        dtype=float
                    )

                    # 先投标过滤构造 candidates
                    candidates = []
                    for i in range(n):
                        tasks = random.sample(range(m), random.choice([1, 2, 3]))
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

                    # PV-DC：基于信任阈值过滤候选
                    candidates = [
                        w for w in candidates
                        if worker_estimated_quality[w['id']] >= q_thre
                    ]

                    if len(candidates) == 0:
                        global_record_times += 1
                        mae_per_round.append(np.nan)
                        quality_history[:, t + 1] = worker_estimated_quality
                        continue

                    workers_ori = copy.deepcopy(candidates)
                    selected_workers = select_workers(candidates, B, m, Q_j)

                    if len(selected_workers) == 0:
                        global_record_times += 1
                        mae_per_round.append(np.nan)
                        quality_history[:, t + 1] = worker_estimated_quality
                        continue

                    payments = compute_payment(selected_workers, workers_ori, B, m, Q_j)
                    _ = sum(payments.values())

                    # 上报数据：高斯噪声 + AQI裁剪
                    submitted_data = []
                    submitted_task_ids = []
                    worker_indices = []
                    selected_worker_ids = [w['id'] for w in selected_workers]

                    for worker in selected_workers:
                        worker_id = worker['id']
                        sigma2 = worker['sigma2']
                        for task_id in worker['tasks']:
                            tv = float(true_task_values[task_id])
                            noise = np.random.normal(0.0, np.sqrt(sigma2))
                            reported = float(np.clip(tv + noise, AQI_MIN, AQI_MAX))

                            submitted_data.append(reported)
                            submitted_task_ids.append(int(task_id))
                            worker_indices.append(int(worker_id))

                    submitted_data = np.array(submitted_data, dtype=float)
                    submitted_task_ids = np.array(submitted_task_ids, dtype=int)
                    worker_indices = np.array(worker_indices, dtype=int)

                    # ========== CRH / TD ==========
                    quality_for_td = worker_estimated_quality[worker_indices].copy()

                    estimated_task_values = np.full(m, np.nan, dtype=float)
                    for j in range(m):
                        idx = np.where(submitted_task_ids == j)[0]
                        if len(idx) > 0:
                            wj = quality_for_td[idx]
                            dj = submitted_data[idx]
                            estimated_task_values[j] = np.sum(wj * dj) / (np.sum(wj) + 1e-6)

                    for _iter_td in range(max_inner_iter):
                        prev_est = estimated_task_values.copy()

                        distance = np.abs(submitted_data - estimated_task_values[submitted_task_ids])
                        quality_for_td = np.exp(-(-np.log(0.5) / 5) * distance)
                        quality_for_td = (quality_for_td - quality_for_td.min()) / (
                            quality_for_td.max() - quality_for_td.min() + 1e-8
                        )

                        for j in range(m):
                            idx = np.where(submitted_task_ids == j)[0]
                            if len(idx) > 0:
                                wj = quality_for_td[idx]
                                dj = submitted_data[idx]
                                estimated_task_values[j] = np.sum(wj * dj) / (np.sum(wj) + 1e-6)
                            else:
                                estimated_task_values[j] = np.nan

                        if np.linalg.norm(estimated_task_values - prev_est, ord=1) < epsilon_td:
                            break

                    # MAE
                    mae = np.nanmean(np.abs(estimated_task_values - true_task_values))
                    mae_per_round.append(mae)

                    # ========== 信任更新 ==========
                    estimated_errors = np.abs(submitted_data - estimated_task_values[submitted_task_ids])
                    z_scores = error_threshold - estimated_errors
                    S_i_all = np.tanh(z_scores)

                    S_i_sum = np.zeros(n, dtype=float)
                    S_i_count = np.zeros(n, dtype=float)
                    for k, wid in enumerate(worker_indices):
                        S_i_sum[wid] += S_i_all[k]
                        S_i_count[wid] += 1.0
                    S_i_final = S_i_sum / (S_i_count + 1e-6)

                    worker_estimated_quality[selected_worker_ids] = (
                        np.exp(-xi) * worker_estimated_quality[selected_worker_ids]
                        + S_i_final[selected_worker_ids] * delta_q
                    )

                    q_min = worker_estimated_quality.min()
                    q_max = worker_estimated_quality.max()
                    worker_estimated_quality = (worker_estimated_quality - q_min) / (q_max - q_min + 1e-6)

                    quality_history[:, t + 1] = worker_estimated_quality

                # execution聚合
                final_mae += np.array(mae_per_round, dtype=float)
                final_quality_normal += np.mean(quality_history[np.where(worker_types == 'normal')[0], :], axis=0)
                final_quality_malicious += np.mean(quality_history[np.where(worker_types == 'malicious')[0], :], axis=0)

            # 平均并保存
            final_mae /= num_executions
            final_quality_normal /= num_executions
            final_quality_malicious /= num_executions

            out_name = f"TE(AQI)-our-PVDC-Q{Q_j}-nor{normal_ratio:.1f}-qthre{q_thre:.1f}.xlsx"
            with pd.ExcelWriter(out_name) as writer:
                pd.DataFrame({"Round": range(num_rounds), "Average MAE": final_mae}).to_excel(
                    writer, sheet_name="MAE", index=False
                )
                pd.DataFrame({
                    "Round": range(num_rounds + 1),
                    "Normal Average Quality": final_quality_normal,
                    "Malicious Average Quality": final_quality_malicious
                }).to_excel(writer, sheet_name="Average Quality", index=False)
                pd.DataFrame([{
                    "Q_j": Q_j,
                    "normal_ratio": normal_ratio,
                    "q_thre": q_thre,
                    "n": n, "m": m, "B": B,
                    "num_rounds": num_rounds,
                    "num_executions": num_executions,
                    "global_record_times": global_record_times,
                    "excel_source": excel_file
                }]).to_excel(writer, sheet_name="Config", index=False)

            print(f"Saved: {out_name}")