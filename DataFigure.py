from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_numbers_from_txt(txt_path: Path) -> np.ndarray:
    s = txt_path.read_text(encoding="utf-8", errors="ignore")
    arr = np.fromstring(s, sep=" ", dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0]
    return arr


def main(base_dir: Path):
    routes = [f"Route {i}" for i in range(1, 7)]
    files = ["A.txt", "B.txt", "C.txt", "D.txt", "E.txt", "F.txt"]

    route_data = {}   # dict: route -> 1D np.array
    summary_rows = []

    # 读取并合并
    for r in routes:
        route_dir = base_dir / r
        if not route_dir.exists():
            print(f"[WARN] not found: {route_dir}")
            route_data[r] = np.array([], dtype=float)
            continue

        all_vals = []
        for fn in files:
            p = route_dir / fn
            if not p.exists():
                print(f"[WARN] not found: {p}")
                continue
            vals = read_numbers_from_txt(p)
            if vals.size > 0:
                all_vals.append(vals)

        merged = np.concatenate(all_vals) if all_vals else np.array([], dtype=float)
        route_data[r] = merged

        summary_rows.append({
            "Route": r,
            "n": int(merged.size),
            "min": float(np.nanmin(merged)) if merged.size else np.nan,
            "q1": float(np.nanpercentile(merged, 25)) if merged.size else np.nan,
            "median": float(np.nanmedian(merged)) if merged.size else np.nan,
            "q3": float(np.nanpercentile(merged, 75)) if merged.size else np.nan,
            "max": float(np.nanmax(merged)) if merged.size else np.nan,
            "mean": float(np.nanmean(merged)) if merged.size else np.nan,
        })

    # -------- 1) 导出到 Excel --------
    # 每个Route一列，长度不一用 NaN 补齐
    df_routes = pd.DataFrame({r: pd.Series(route_data[r]) for r in routes})
    df_summary = pd.DataFrame(summary_rows)

    excel_path = base_dir / "routes_merged_data.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_routes.to_excel(writer, sheet_name="data", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"[OK] Excel saved: {excel_path}")

    # -------- 2) （可选）继续画6个Route对比箱线图并保存 --------
    plot_data = [route_data[r] if route_data[r].size > 0 else np.array([np.nan]) for r in routes]
    sample_sizes = [int(route_data[r].size) for r in routes]

    plt.figure(figsize=(10, 5), dpi=150)
    plt.boxplot(plot_data, tick_labels=routes, showmeans=True)
    plt.title("Boxplot Comparison of 6 Routes (A~F merged, zeros removed)")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    ymax = plt.ylim()[1]
    for i, n in enumerate(sample_sizes, start=1):
        plt.text(i, ymax, f"n={n}", ha="center", va="top", fontsize=9)

    fig_path = base_dir / "boxplots_routes_compare.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[OK] Figure saved: {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main(Path.cwd())