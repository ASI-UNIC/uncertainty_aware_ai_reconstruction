import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
import os


data_plots = [
    "case 0",
    "case 1",
    "case 2",
    "case 3",
]

case0_sparsity = [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64]
case0_seq = [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32]
case0_mape = [0.62, 4.18, 0.39, 0.50, 0.73, 9.84, 0.71, 1.35, 0.58, 1.94, 1.03, 0.55, 1.87, 7.27, 1.64, 0.93, 10.49, 8.78, 8.22, 4.55, 10.58, 11.94, 16.50, 10.70]

case1_sparsity = [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64]
case1_seq = [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32]
case1_mape = [1.06, 0.78, 0.65, 0.54, 2.63, 12.33, 0.35, 0.63, 1.68, 1.44, 0.87, 1.06, 1.76, 3.55, 1.54, 3.58, 16.78, 17.62, 6.13, 7.54, 14.35, 7.2, 7.57, 6.96]

case2_sparsity = [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64]
case2_seq = [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32]
case2_mape = [0.55, 8.95, 3.52, 0.13, 0.36, 1.17, 2.1, 0.54, 2.63, 7.25, 4.98, 0.7, 3.89, 0.8, 2.05, 1.7, 21.56, 15.65, 7.05, 8.09, 20.05, 9.43, 7.93, 9.38]

case3_sparsity = [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64]
case3_seq = [4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32, 4, 8, 16, 32]
case3_mape = [0.83, 0.82, 0.22, 0.22, 1.12, 0.66, 0.79, 0.6, 1.05, 4.13, 1.56, 0.93, 0.99, 1.3, 1.55, 3.61, 18.66, 22.35, 7.64, 6.79, 19.47, 12.74, 12.7, 17.95]

cases_sparsity = [
    case0_sparsity,
    case1_sparsity,
    case2_sparsity,
    case3_sparsity,
]
cases_seq = [
    case0_seq,
    case1_seq,
    case2_seq,
    case3_seq,
]
cases_mape = [
    case0_mape,
    case1_mape,
    case2_mape,
    case3_mape,
]




for index in range(len(data_plots)):
    data_name = data_plots[index]
    folder_dir = f"results/{data_name}/"
    plot_dir = f"results/{data_name}/3d/"
    os.makedirs(plot_dir, exist_ok=True)


    # Example data
    
    # Flatten data for interpolation
    x = cases_sparsity[index]
    y = cases_seq[index]
    z = cases_mape[index]

    # Interpolation grid
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
    Zi = np.maximum(Zi, 0)


    global_vmin = np.min(Zi)
    global_vmax = np.max(Zi)

    # ---------------------------
    # Figure 1: 3D surface plot
    # ---------------------------
    font = {
        'weight' : 'normal',
        'size'   : 22}

    matplotlib.rc('font', **font)
    fig1 = plt.figure(figsize=(8, 8), constrained_layout=True)
    ax1 = fig1.add_subplot(111, projection="3d")
    surf = ax1.plot_surface(Xi, Yi, Zi, cmap="viridis", edgecolor="none", alpha=0.9, vmin=0, vmax=global_vmax)
    ax1.set_xlabel("Sparsity", labelpad=20)
    ax1.set_ylabel("Sequence Length", labelpad=20)
    ax1.zaxis.set_tick_params(pad=10)
    # ax1.set_zlabel("MAPE", labelpad=5)
    
    # Get axis limits
    zmax = ax1.get_zlim()[1] + 2
    xmax = ax1.get_xlim()[1] - 1
    ymax = ax1.get_ylim()[1] - 1

    # Place label at top of Z axis
    ax1.text(xmax, ymax, zmax, "MAPE")
    # ax1.set_title("3D Surface: MAPE vs Layers & Neurons")

    cbar = fig1.colorbar(surf, shrink=0.5, aspect=8, pad=0.1)
    plt.savefig(os.path.join(plot_dir, "MAPE_3D_Surface.png"), dpi=600)
    plt.close(fig1)

    # ---------------------------
    # Figure 2: 2D heatmap (contour)
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    levels = np.linspace(global_vmin, global_vmax, 21)
    contour = ax2.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", vmin=0, vmax=global_vmax)
    cbar2 = fig2.colorbar(contour, label="MAPE")
    cbar2.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_xlabel("Sparsity", fontsize=24)
    ax2.set_ylabel("Sequence Length", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "MAPE_Heatmap.png"), dpi=600)
    plt.close(fig2)

    # ---------------------------
    # Figure 3: 3D scatter plot (non-interpolated)
    # ---------------------------
    fig3 = plt.figure(figsize=(8, 8), constrained_layout=True)
    ax3 = fig3.add_subplot(111, projection="3d")
    scatter3d = ax3.scatter(x, y, z, c=z, cmap="viridis", s=100, vmin=0, vmax=global_vmax, alpha=0.9)
    ax3.set_xlabel("Sparsity", labelpad=20)
    ax3.set_ylabel("Sequence Length", labelpad=20)
    ax3.zaxis.set_tick_params(pad=10)
    
    # Get axis limits for label placement
    zmax3 = ax3.get_zlim()[1] + 2
    xmax3 = ax3.get_xlim()[1] - 1
    ymax3 = ax3.get_ylim()[1] - 1

    # Place label at top of Z axis
    ax3.text(xmax3, ymax3, zmax3, "MAPE")

    cbar3 = fig3.colorbar(scatter3d, shrink=0.5, aspect=8, pad=0.1)
    plt.savefig(os.path.join(plot_dir, "MAPE_3D_Scatter.png"), dpi=600)
    plt.close(fig3)

    # ---------------------------
    # Figure 4: 2D scatter plot (non-interpolated)
    # ---------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    scatter2d = ax4.scatter(x, y, c=z, cmap="viridis", s=150, vmin=0, vmax=global_vmax, alpha=0.9)
    cbar4 = fig4.colorbar(scatter2d, label="MAPE")
    cbar4.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax4.set_xlabel("Sparsity", fontsize=24)
    ax4.set_ylabel("Sequence Length", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "MAPE_2D_Scatter.png"), dpi=600)
    plt.close(fig4)
