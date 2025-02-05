import numpy as np
import os
import pywt
import matplotlib.pyplot as plt
import string

# 设置全局字体大小和字体
plt.rcParams.update({
    "font.size": 26,  # 统一字体大小
    "axes.titlesize": 26,  # 坐标轴标题字体大小
    "axes.labelsize": 26,  # 坐标轴标签字体大小
    "legend.fontsize": 26,  # 图例字体大小
    # "font.family": "serif",  # 使用衬线字体
})

# 数据加载
data_dir = "../data/PEMS08"
data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

flow_data = data[..., 0]  # (time_steps, num_nodes)
sensor_flow_data = flow_data[3000:8000, 0]  # (time_steps)

# 小波变换参数
wavelet = 'db2'  # 使用Daubechies 2小波
level = 4  # 设定分解的层数

# 对交通流数据进行小波分解
coeffs = pywt.wavedec(sensor_flow_data, wavelet, level=level)

# 绘制时频特征图
fig, axes = plt.subplots(level, 1, figsize=(16, 12), sharex=True)  # 不绘制cA1
# fig, axes = plt.subplots(level + 1, 1, figsize=(16, 12), sharex=True)
# fig.suptitle("Wavelet Transform Decomposition", fontsize=24)

# 原始信号
axes[0].plot(sensor_flow_data, linewidth=1.5)
axes[0].set_title('(a) Original Signal', loc='left')
axes[0].grid(True, color="gray", linestyle='--', alpha=0.6)

ax_idx = 1  # 用于指定从第二个图开始绘制
for i in range(level):

    # 跳过cA1
    if i == level - 1:
        continue

    # 为当前层的恢复创建一个系数列表，其他层用零代替
    current_coeffs = []
    for j, coef in enumerate(coeffs):
        if j == i:
            current_coeffs.append(coef)  # 当前层的系数
        else:
            current_coeffs.append(np.zeros_like(coef))  # 其他层设为零


    # 使用逆小波变换恢复信号
    reconstructed_signal = pywt.waverec(current_coeffs, wavelet)

    # 使用字母编号
    title = f'({string.ascii_lowercase[ax_idx]}) cA{level - i} Reconstructed Signal'
    axes[ax_idx].plot(reconstructed_signal[:len(sensor_flow_data)], linewidth=1.5)
    axes[ax_idx].set_title(title, loc='left')
    axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
    ax_idx += 1


# 设置共享横坐标标签
axes[-1].set_xlabel("Time Step")
axes[-1].set_xlim(0, len(sensor_flow_data))

fig.text(0.02, 0.5, 'Flow', ha='center', va='center', rotation='vertical')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("wavelet.pdf", dpi=300, bbox_inches='tight')
plt.show()
