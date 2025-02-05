import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 设置Seaborn样式，使图表更加美观
sns.set(style="whitegrid", palette="muted")
# 设置全局字体大小和字体
plt.rcParams.update({
    "font.size": 26,  # 统一字体大小
    "axes.titlesize": 30,  # 坐标轴标题字体大小
    "axes.labelsize": 30,  # 坐标轴标签字体大小
    "legend.fontsize": 24,  # 图例字体大小
})

# 手动填入指标数据
data = {
    "Datasets": ["PEMS03", "PEMS04", "PEMS07", "PEMS08"],
    "Model": ["M1", "M2", "M3", "M4"],
    "RMSE": [25.62, 25.63, 26.47, 25.25,
             30.65, 30.72, 30.65, 30.40,
             32.67, 32.82, 32.61, 32.85,
             22.74, 22.72, 23.24, 23.34],
    "MAE": [14.96, 15.00, 15.33, 15.08,
            18.17, 18.30, 18.30, 18.19,
            19.10, 19.51, 19.15, 19.29,
            13.14, 13.21, 13.43, 13.47],
    "MAPE": [15.07, 15.22, 15.33, 15.21,
             11.97, 12.05, 12.01, 12.00,
             7.93, 8.33, 7.99, 8.04,
             8.67, 8.73, 8.87, 8.83]
}

# 颜色映射，与其他绘图一致
colors = ['#1f77b4', '#ff7f0e', '#e94e77', '#2ca02c']

# 画分组柱状图（3行，每行包含3个指标）
fig, axes = plt.subplots(len(data["Datasets"]), 3, figsize=(18, 18))
metrics = ["RMSE", "MAE", "MAPE"]

for row, dataset in enumerate(data["Datasets"]):
    for i, (ax, metric) in enumerate(zip(axes[row], metrics)):
        values = np.array(data[metric][row * 4:(row + 1) * 4])
        value_min, value_max = values.min(), values.max()
        value_range = value_max - value_min
        expand = value_range * 0.5  # 计算扩展范围

        ymin = value_min - expand
        ymax = value_max + expand

        bars = ax.bar(data["Model"], values, color=colors, width=0.7)
        ax.set_title(metric)
        ax.set_ylim(ymin, ymax)  # 使用计算得到的y轴范围
        ax.tick_params(axis='both', labelsize=24)  # 设置刻度的字体大小

        # 设置y轴刻度格式为保留两位小数
        formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
        ax.yaxis.set_major_formatter(formatter)

        # 在柱状图上标出数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=20)

        # 在第一列加上数据集名称
        if i == 0:
            ax.set_ylabel(dataset, labelpad=20)

# 仅在第一行添加图例
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(4)]
fig.legend(handles, ["M1", "M2", "M3", "M4"], loc='upper center', ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ablation.svg", dpi=300, bbox_inches='tight')
plt.show()
