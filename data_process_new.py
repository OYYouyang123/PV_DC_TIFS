import numpy as np
import os
import pandas as pd

# --------- 配置 ---------
file_names = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt', 'F.txt']
num_time_slots = 301  # 目标时隙长度（每个数组应为 301）

# 两组 Route：分别输出到两个 Excel
routes_group1 = [f'Route {i}' for i in range(1, 5)]  # Route 1-4 -> excel 1
routes_group2 = [f'Route {i}' for i in range(5, 7)]  # Route 5-6 -> excel 2

excel_file_1 = "output_data_Route1_4.xlsx"
excel_file_2 = "output_data_Route5_6.xlsx"


def fix_length_to_301(arr, target_len=301):
    """
    将一维数组修正到 target_len：
    - 长度不足：用该数组的平均值补足
    - 长度超出：截断
    - 空数组：用 0 补足（避免 mean 为 nan）
    """
    arr = np.asarray(arr).reshape(-1)  # 强制一维

    if arr.size == 0:
        fill_value = 0.0
    else:
        fill_value = float(np.mean(arr))

    if arr.size < target_len:
        pad = np.full(target_len - arr.size, fill_value, dtype=float)
        arr = np.concatenate([arr.astype(float, copy=False), pad])
    elif arr.size > target_len:
        arr = arr[:target_len]

    return arr


def route_number_from_name(route_name: str) -> str:
    # "Route 12" -> "12"，避免使用 route[-1] 在两位数时出错
    return route_name.split()[-1]


def process_routes_to_excel(routes, excel_file):
    # 创建一个字典来存储数组
    data_arrays = {}

    # 遍历文件夹和文件
    for route in routes:
        for file_name in file_names:
            file_path = os.path.join(route, file_name)

            try:
                data = np.loadtxt(file_path)

                # 修正长度到 301（尤其针对 Route 1 的不满足情况）
                data = fix_length_to_301(data, target_len=num_time_slots)

                # 创建数组名称：R{route编号}_{文件首字母}
                rnum = route_number_from_name(route)
                array_name = f'R{rnum}_{file_name[0]}'
                data_arrays[array_name] = data
            except Exception as e:
                print(f'Error reading {file_path}: {e}')

    # 将数据存储到 Excel 文件：每个数组写到单独 sheet，单独一列
    with pd.ExcelWriter(excel_file) as writer:
        for array_name, array_data in data_arrays.items():
            df = pd.DataFrame(array_data, columns=[array_name])
            df.to_excel(writer, sheet_name=array_name, index=False)

    print(f"Data has been saved to {excel_file}")

    # 从 Excel 文件中按时隙取出数据（逻辑不变）
    data_from_excel = pd.read_excel(excel_file, sheet_name=None)

    for time_slot in range(num_time_slots):
        true_task_values = []
        for array_name, df in data_from_excel.items():
            true_task_values.append(df.iloc[time_slot, 0])
        print(f"[{os.path.basename(excel_file)}] Time Slot {time_slot + 1}: {true_task_values}")


# --------- 执行：生成两个 Excel，并按时隙打印 ---------
process_routes_to_excel(routes_group1, excel_file_1)
process_routes_to_excel(routes_group2, excel_file_2)