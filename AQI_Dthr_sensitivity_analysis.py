import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,          # 全局默认字体大小
    'axes.titlesize': 16,     # 子图标题字体大小
    'axes.labelsize': 16,     # 坐标轴标签字体大小
    'xtick.labelsize': 14,    # x轴刻度字体大小
    'ytick.labelsize': 14,    # y轴刻度字体大小
    'legend.fontsize': 16,    # 图例字体大小
    'figure.titlesize': 16    # 整个figure标题字体大小（若使用 suptitle）
})



# ==========================================================
# 参数设置（固定，与AQI主实验保持一致）
# ==========================================================

n = 150
m = 12
B = 300


beta = 0.95

xi = 0.1

delta_q = 0.1

epsilon_td = 1e-4

max_inner_iter = 30


num_rounds = 301

num_executions = 10



# ==========================================================
# 固定实验参数
# ==========================================================

Q_j = 5

normal_ratio = 0.6

q_thre = 0.6



# ==========================================================
# D_thr敏感性分析参数
# ==========================================================

Dthr_list = [

    0.5,

    1,

    2,

    3,

    4,      # AQI默认值

    5,

    6,

    8,

    10

]



# ==========================================================
# AQI数据读取
# ==========================================================

excel_file = "output_data_AQI3_m12_r301_skipNaN.xlsx"


data_from_excel = pd.read_excel(
    excel_file,
    sheet_name=None
)


sheet_names = sorted(
    data_from_excel.keys()
)



AQI_MIN = 0.0
AQI_MAX = 500.0



base_seed = 2026



# ==========================================================
# 保存结果
# ==========================================================

Dthr_results = {}



# ==========================================================
# 不同D_thr实验
# ==========================================================


for error_threshold in Dthr_list:


    print(
        "\nRunning AQI D_thr = ",
        error_threshold
    )


    final_mae = np.zeros(
        num_rounds
    )


    final_quality_normal = np.zeros(
        num_rounds+1
    )


    final_quality_malicious = np.zeros(
        num_rounds+1
    )



    # ======================================================
    # 多次实验
    # ======================================================

    for execution in range(num_executions):


        seed = (
            base_seed
            +
            int(error_threshold*100)
            +
            execution
        )


        np.random.seed(seed)

        random.seed(seed)



        # ==============================
        # worker初始化
        # ==============================

        workers=[]



        num_normal=int(
            normal_ratio*n
        )


        num_malicious=n-num_normal



        worker_types=np.array(
            ["normal"]*num_normal
            +
            ["malicious"]*num_malicious
        )


        np.random.shuffle(
            worker_types
        )



        # AQI噪声模型
        lambda_nor_aqi = 1/2

        lambda_mal_aqi = 1/36



        scales=np.where(
            worker_types=="normal",
            1/lambda_nor_aqi,
            1/lambda_mal_aqi
        )



        worker_sigma2=np.random.exponential(
            scale=scales
        )



        worker_quality=np.ones(n)



        quality_history=np.zeros(
            (n,num_rounds+1)
        )


        quality_history[:,0]=1



        mae_history=[]




        for i in range(n):


            workers.append({

                "id":i,

                "type":
                worker_types[i],

                "sigma2":
                worker_sigma2[i],

                "q_i":1,

                "w_loss":1

            })




        # ==================================================
        # round循环
        # ==================================================

        for t in range(num_rounds):


            true_task_values=np.array(

                [
                    data_from_excel[name]
                    .iloc[t,0]

                    for name in sheet_names

                ],

                dtype=float

            )



            # ---------------------------------
            # candidate worker
            # ---------------------------------

            candidates=[]


            for i,w in enumerate(workers):


                tasks=random.sample(
                    range(m),
                    random.choice([1,2,3])
                )


                w["tasks"]=tasks


                w["b_i"]=len(tasks)


                if worker_quality[i]>=q_thre:

                    candidates.append(w)




            if len(candidates)==0:


                mae_history.append(
                    np.nan
                )

                continue




            # 简化选择
            selected_workers=candidates




            submitted_data=[]

            task_ids=[]

            worker_ids=[]




            for w in selected_workers:


                for task in w["tasks"]:


                    noise=np.random.normal(

                        0,

                        np.sqrt(
                            w["sigma2"]
                        )

                    )


                    reported=np.clip(

                        true_task_values[task]
                        +
                        noise,

                        AQI_MIN,

                        AQI_MAX

                    )


                    submitted_data.append(
                        reported
                    )


                    task_ids.append(
                        task
                    )


                    worker_ids.append(
                        w["id"]
                    )




            submitted_data=np.array(
                submitted_data
            )


            task_ids=np.array(
                task_ids
            )


            worker_ids=np.array(
                worker_ids
            )




            # ==================================================
            # Truth Discovery
            # ==================================================


            estimate=np.zeros(m)



            for j in range(m):


                idx=np.where(
                    task_ids==j
                )[0]


                if len(idx)>0:


                    estimate[j]=np.mean(
                        submitted_data[idx]
                    )




            for _ in range(max_inner_iter):


                old=estimate.copy()



                distance=np.abs(

                    submitted_data
                    -
                    estimate[task_ids]

                )



                weight=np.exp(

                    -0.1386
                    *
                    distance

                )



                for j in range(m):


                    idx=np.where(
                        task_ids==j
                    )[0]


                    if len(idx)>0:


                        estimate[j]=(

                            np.sum(
                                weight[idx]
                                *
                                submitted_data[idx]
                            )

                            /

                            np.sum(
                                weight[idx]
                            )

                        )



                if np.linalg.norm(
                    estimate-old
                ) < epsilon_td:

                    break





            # MAE

            mae=np.mean(

                np.abs(

                    estimate
                    -
                    true_task_values

                )

            )


            mae_history.append(
                mae
            )




            # ==================================================
            # Trust update
            # ==================================================


            errors=np.abs(

                submitted_data
                -
                estimate[task_ids]

            )



            # D_thr核心
            score=np.tanh(

                error_threshold
                -
                errors

            )



            for wid in np.unique(worker_ids):


                worker_quality[wid]=(

                    np.exp(-xi)
                    *
                    worker_quality[wid]

                    +

                    delta_q
                    *
                    np.mean(

                        score[
                            worker_ids==wid
                        ]

                    )

                )




            # normalize

            worker_quality=(

                worker_quality
                -
                worker_quality.min()

            )/(

                worker_quality.max()
                -
                worker_quality.min()
                +
                1e-6

            )



            quality_history[:,t+1]=worker_quality





        # ==================================================
        # execution结果累积
        # ==================================================

        final_mae += np.array(
            mae_history
        )


        final_quality_normal += np.mean(

            quality_history[
                worker_types=="normal"
            ],

            axis=0

        )


        final_quality_malicious += np.mean(

            quality_history[
                worker_types=="malicious"
            ],

            axis=0

        )




    # ==================================================
    # 平均
    # ==================================================

    final_mae/=num_executions


    final_quality_normal/=num_executions


    final_quality_malicious/=num_executions




    Dthr_results[error_threshold]={


        "MAE":
        final_mae,


        "normal":
        final_quality_normal,


        "malicious":
        final_quality_malicious

    }





# ==========================================================
# Figure 1
# Dthr-MAE
# ==========================================================


mean_mae=[]


for d in Dthr_list:


    mean_mae.append(

        np.nanmean(

            Dthr_results[d]["MAE"][-50:]

        )

    )



plt.figure(figsize=(8,5))


plt.plot(

    Dthr_list,

    mean_mae,

    marker="o"

)



plt.xlabel(
    "$D_{thr}$"
)


plt.ylabel(
    "Final MAE"
)



plt.grid()


plt.title(
    "Sensitivity analysis of $D_{thr}$ in AQI Monitoring"
)



plt.savefig(

    "AQI_Dthr_MAE.png",

    dpi=300,

    bbox_inches="tight"

)



plt.show()




# ==========================================================
# Figure 2
# 不同Dthr收敛曲线
# ==========================================================


plt.figure(
    figsize=(8,5)
)



for d in Dthr_list:


    plt.plot(

        Dthr_results[d]["MAE"],

        label=f"D={d}"

    )



plt.xlabel(
    "Round"
)



plt.ylabel(
    "MAE"
)



plt.legend()


plt.grid()



plt.title(
    "MAE convergence under different $D_{thr}$ in AQI Monitoring"
)



plt.savefig(

    "AQI_Dthr_convergence.png",

    dpi=300,

    bbox_inches="tight"

)



plt.show()




# ==========================================================
# Figure 3
# Trust separation
# ==========================================================


plt.figure(
    figsize=(8,5)
)



for d in [

    0.5,

    4,

    10

]:


    gap=(

        Dthr_results[d]["normal"]

        -

        Dthr_results[d]["malicious"]

    )



    plt.plot(

        gap,

        label=f"D={d}"

    )



plt.xlabel(
    "Round"
)


plt.ylabel(
    "Trust gap"
)



plt.legend()


plt.grid()



plt.title(
    "Trust discrimination under different $D_{thr}$ in AQI Monitoring"
)



plt.savefig(

    "AQI_Dthr_trust_gap.png",

    dpi=300,

    bbox_inches="tight"

)


plt.show()