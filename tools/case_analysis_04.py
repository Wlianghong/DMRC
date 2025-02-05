import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# 设置Seaborn样式，使图表更加美观
sns.set(style="whitegrid", palette="muted")
# 设置全局字体大小和字体
plt.rcParams.update({
    "font.size": 26,  # 统一字体大小
    "axes.titlesize": 30,  # 坐标轴标题字体大小
    "axes.labelsize": 26,  # 坐标轴标签字体大小
    "legend.fontsize": 22,  # 图例字体大小
})

line_width = 4

# 日志文件目录和文件名
log_dir = "../logs"
file1 = "DMRCMLP-PEMS04-2025-01-16-16-26-55.log"  # DMRC
file2 = "DMRCMLP-PEMS04-2025-01-17-18-56-24.log"  # without reconstruction, and without masking

# 定义正则表达式来匹配目标行
pattern = r"Epoch (\d+), Train Y Loss = ([\d.]+),  Train X Loss = ([\d.]+), Val Loss = ([\d.]+)"

# 提取日志文件中的损失数据
def extract_loss_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    matches = re.findall(pattern, log_content)

    epochs = [int(match[0]) for match in matches]
    train_y_losses = [float(match[1]) for match in matches]
    train_x_losses = [float(match[2]) for match in matches]
    val_losses = [float(match[3]) for match in matches]

    return epochs, train_y_losses, train_x_losses, val_losses


# 提取文件1和文件2的数据
epochs1, train_y_losses1, train_x_losses1, val_losses1 = extract_loss_data(os.path.join(log_dir, file1))
epochs2, train_y_losses2, train_x_losses2, val_losses2 = extract_loss_data(os.path.join(log_dir, file2))

# 创建一个图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制文件1的Y Loss、X Loss和Val Loss曲线
ax.plot(epochs1, train_y_losses1, label="Pred. Train Loss (M1)", color='#1f77b4', linestyle=':', linewidth=line_width)
ax.plot(epochs1, train_x_losses1, label="Rec. Train Loss (M1)", color='#ff7f0e', linestyle='-.', linewidth=line_width)
ax.plot(epochs1, val_losses1, label="Pred. Val Loss (M1)", color='#e94e77', linestyle='-', linewidth=line_width)

# 绘制文件2的Validation Loss曲线
ax.plot(epochs2, val_losses2, label="Pred. Val Loss (M4)", color='#2ca02c', linestyle='--', linewidth=line_width)


# 添加标题和标签
ax.set_title("PEMS04")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss")

# 调整坐标轴标签字体
ax.tick_params(axis='both', which='major', labelsize=20)

# 使网格线更加细致
ax.grid(True, linestyle='--', alpha=0.7)

# 显示图例并自定义位置
# ax.legend(loc='upper right')
ax.legend(loc='lower left')

# 设置主图坐标轴边框颜色为深灰色
# ax.spines['top'].set_color('#4b4b4b')   # 深灰色
ax.spines['bottom'].set_color('#4b4b4b')
ax.spines['left'].set_color('#4b4b4b')
# ax.spines['right'].set_color('#4b4b4b')

# 设置主图坐标轴边框的粗细
# ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)


##################################################################################
x_start = 20
x_end = 60
y_start = 17
y_end = 18

# 创建一个 GridSpec 对象，控制子图的网格位置
gs = gridspec.GridSpec(1, 1, left=0.5, right=0.9, top=0.88, bottom=0.66)  # 这里调整 left 和 top 参数来控制子图位置

# 通过 GridSpec 创建子图
axins = fig.add_subplot(gs[0])

# 选择显示的区间（20-60 epoch，val_loss在13-14之间）
zoomed_epochs1 = [e for e in epochs1 if x_start <= e <= x_end]
zoomed_val_losses1 = [val_losses1[i] for i, e in enumerate(epochs1) if x_start <= e <= x_end]
zoomed_epochs2 = [e for e in epochs2 if x_start <= e <= x_end]
zoomed_val_losses2 = [val_losses2[i] for i, e in enumerate(epochs2) if x_start <= e <= x_end]

# 在子图中绘制放大的val_loss曲线
axins.plot(zoomed_epochs1, zoomed_val_losses1, color='#e94e77', linestyle='-', linewidth=5)
axins.plot(zoomed_epochs2, zoomed_val_losses2, color='#2ca02c', linestyle='--', linewidth=5)

# 设置子图的坐标轴范围
axins.set_xlim(x_start, x_end)
axins.set_ylim(y_start, y_end)

# 去除子图的图例
axins.legend().set_visible(False)

# 设置子图边框的颜色为深灰色
axins.spines['top'].set_color('#4b4b4b')   # 深灰色
axins.spines['bottom'].set_color('#4b4b4b')
axins.spines['left'].set_color('#4b4b4b')
axins.spines['right'].set_color('#4b4b4b')

# 设置子图边框的粗细
axins.spines['top'].set_linewidth(2)
axins.spines['bottom'].set_linewidth(2)
axins.spines['left'].set_linewidth(2)
axins.spines['right'].set_linewidth(2)



# 展示图表
plt.tight_layout()  # 调整布局以避免标签被遮挡
plt.savefig("case_study_pems04.svg", dpi=300, bbox_inches='tight')
plt.show()
