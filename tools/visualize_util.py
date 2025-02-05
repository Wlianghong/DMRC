import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# 设置全局字体大小和字体
plt.rcParams.update({
    "font.size": 16,  # 统一字体大小
    "axes.titlesize": 20,  # 坐标轴标题字体大小
    "axes.labelsize": 16,  # 坐标轴标签字体大小
    "legend.fontsize": 16,  # 图例字体大小
})


# PCA 降维
def pca_reduction(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# t-SNE 降维
def tsne_reduction(data, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=2)
    return tsne.fit_transform(data)

# UMAP 降维
def umap_reduction(data, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=2)
    return umap_model.fit_transform(data)

# 3D 可视化
def plot_3d(data, colors, title="3D Projection", show_title=True, save=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap='viridis', s=10)

    # 设置视角
    # ax.view_init(elev=30, azim=-45)  # 30度俯仰角，45度方位角

    if show_title:
        plt.title(title, loc='center')
    # plt.colorbar(sc)  # 显示颜色条
    if save:
        plt.savefig(f"{title}.pdf", format="pdf", dpi=300)
    plt.show()

# 2D 可视化
def plot_2d(data, colors, title="2D Projection", show_title=True, save=True):
    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='viridis', s=10)
    if show_title:
        plt.title(title, loc='center')
    # plt.colorbar()  # 显示颜色条
    if save:
        plt.savefig(f"{title}.svg", format="svg", dpi=300)
    plt.show()

def visualize(data, method="umap", dim=2, title="", show_title=True, save=True):
    # 可以选择基于数据的某个维度或者所有维度的均值作为颜色值
    colors = np.linalg.norm(data, axis=1)  # 使用每个点的L2范数作为颜色值（可以根据需求调整）

    if method == 'pca':
        reduced_data = pca_reduction(data, n_components=dim)
    elif method == 'tsne':
        reduced_data = tsne_reduction(data, n_components=dim)
    elif method == 'umap':
        reduced_data = umap_reduction(data, n_components=dim)
    else:
        print("无效的方法，使用默认PCA降维。")
        reduced_data = pca_reduction(data, n_components=dim)

    if dim == 2:
        plot_2d(reduced_data, colors, title, show_title, save)
    elif dim == 3:
        plot_3d(reduced_data, colors, title, show_title, save)
    else:
        print("不支持的可视化维度")

if __name__ == "__main__":
    # 创建一个随机的高维数据集
    data = np.random.rand(100, 50)  # 100个样本，50维特征
    visualize(data)
