import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from objects.obstacle.objs.obstacle_clip_gt import ObstacleClipGt
from objects.obstacle.parsers.attribute_tool import Attr
from utils.plot import polar_histogram, histogram_2d_sns, polar_histogram_core


def wrap_yaw(yaw):
    if yaw < 0:
        return yaw + 2 * np.pi
    return yaw


def get_pos_angle(points):
    points[:, 1] *= -1
    direction = points / np.linalg.norm(points)
    radians = np.arctan2(direction[:, 1].astype(np.float64), direction[:, 0].astype(np.float64))
    radians[radians < 0] += 2 * np.pi
    return radians


def set_ax_xticks(ax, func=lambda x: "{}m".format(x)):
    x_values = ax.get_xticks()
    x_labels = [func(value) for value in x_values]
    ax.set_xticks(x_values, x_labels)


def plot_data_description(title, df_data, out_path=None):
    """
    provide
    1. histogram over length
    2. histogram over width
    3. histogram over height
    4. histogram over distance
    5. histogram over orientation
    6. histogram over number of objs per frame
    7. histogram in polar coordination over radius and position angle
    """
    print("process {}".format(title))
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    gs = GridSpec(6, 12, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :3])
    ax2 = fig.add_subplot(gs[2:4, :3])
    ax3 = fig.add_subplot(gs[4:6, :3])
    ax4 = fig.add_subplot(gs[:2, 3:7])
    ax5 = fig.add_subplot(gs[2:4, 3:7])
    ax6 = fig.add_subplot(gs[4:, 3:7])
    ax7 = fig.add_subplot(gs[:, 7:], projection="polar")

    fig.suptitle(title)
    pc = sns.histplot(df_data, x=Attr.length, edgecolor="white", ax=ax1, bins=30, color="#F69E06")
    set_ax_xticks(ax1, func=lambda x: "{:.2f}m".format(x))

    sns.histplot(df_data, x=Attr.width, edgecolor="white", ax=ax2, bins=30, color="#4286F4")
    set_ax_xticks(ax2, func=lambda x: "{:.2f}m".format(x))
    sns.histplot(df_data, x=Attr.height, edgecolor="white", ax=ax3, bins=30, color="#F23D64")
    set_ax_xticks(ax3, func=lambda x: "{:.2f}m".format(x))
    sns.histplot(df_data, x="Distance", edgecolor="white", ax=ax4, bins=300)
    set_ax_xticks(ax4, func=lambda x: "{:.0f}m".format(x))
    sns.histplot(df_data, x="orientation", edgecolor="white", ax=ax5, bins=360)
    set_ax_xticks(ax5, func=lambda x: "{:.0f}°".format(x))

    frame_num_count = []
    for ts, group_data in df_data.groupby([Attr.frame_seq]):
        frame_num_count.append(group_data.shape[0])
    frame_num_data = pd.DataFrame({"objs_in_frame": frame_num_count})
    sns.histplot(frame_num_data, x="objs_in_frame", edgecolor="white", ax=ax6)
    ax6.set_title("average: {}".format(int(frame_num_data["objs_in_frame"].mean())), y=1.1, fontsize=13)

    # sns.histplot(df_data, x="Distance", y="Lidar points in box",
    #              bins=20, discrete=(False, False), log_scale=(False, True),
    #              ax=ax6, cmap="Blues", cbar=True)
    # sns.histplot(df_data, x="Distance", y="Lidar points in box",
    #              bins=20, discrete=(False, False), log_scale=(False, True),
    #              ax=ax6, cbar=True)
    # ax6.set_aspect(30)
    # set_ax_xticks(ax6, func=lambda x: "{:.0f}m".format(x))
    polar_histogram_core(df_data["pos_angle"].to_numpy(),
                         df_data["Distance"].to_numpy(),
                         fig,
                         ax7,
                         rbin_num=6,
                         abin_num=9,
                         title="total: {}".format(len(df_data["pos_angle"].tolist())))

    if out_path:
        fig.set_dpi(300)
        out_figure_path = os.path.join(out_path, "{}.png".format(title))
        plt.savefig(out_figure_path)
    else:
        plt.show()
    plt.close(fig)


def histogram_on_each_category(df_data, out_path=None):
    df_data["orientation"] = np.degrees(df_data[Attr.lidar_yaw].to_numpy())
    df_data["Lidar points in box"] = df_data[Attr.num_lidar_pts]
    # df_data = df_data[df_data["Lidar points in box"] > 0]
    df_data["Distance"] = np.sqrt((df_data[Attr.lidar_pos_x] ** 2 + df_data[Attr.lidar_pos_y] ** 2).astype(float))
    df_data["pos_angle"] = get_pos_angle(df_data[[Attr.lidar_pos_x, Attr.lidar_pos_y]].to_numpy())
    df_data = df_data[df_data["Distance"] <= 150]
    for category, group_data in df_data.groupby(["category"]):
        plot_data_description(category, group_data, out_path)
    plot_data_description("Overall", df_data, out_path)


def num_histogram(df_data, out_path=None):
    df_data["orientation"] = np.degrees(df_data[Attr.lidar_yaw].to_numpy())
    df_data["Lidar points in box"] = df_data[Attr.num_lidar_pts]
    # df_data = df_data[df_data["Lidar points in box"] > 0]
    df_data["Distance"] = np.sqrt((df_data[Attr.lidar_pos_x] ** 2 + df_data[Attr.lidar_pos_y] ** 2).astype(float))
    df_data["pos_angle"] = get_pos_angle(df_data[[Attr.lidar_pos_x, Attr.lidar_pos_y]].to_numpy())
    df_data = df_data[df_data["Distance"] <= 150]

    total_num = df_data.shape[0]
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    sns.histplot(df_data, x="category", edgecolor='white', stat="percent", ax=ax1, bins=300)
    y_values = ax1.get_yticks()
    y_labels = ["{}%".format(value) for value in y_values]
    ax1.set_yticks(y_values, y_labels)
    for container in ax1.containers:
        ax1.bar_label(container, fmt=lambda x: int(total_num * x / 100))
        # ax1.bar_label(container)
    ax1.set_title("total: {}".format(total_num), y=1.1, fontsize=13)
    sns.histplot(df_data, x="Distance", hue="category", edgecolor="white", ax=ax2, multiple="dodge", bins=15, alpha=0.5)
    set_ax_xticks(ax2)
    if out_path is not None:
        fig.set_dpi(300)
        out_figure_path = os.path.join(out_path, "all_category_stat.png")
        plt.savefig(out_figure_path)
    else:
        plt.show()
    plt.close(fig)


def data_statistics(df_data, out_path):
    df_data["Lidar points in box"] = df_data[Attr.num_lidar_pts]
    df_data["Distance"] = np.sqrt((df_data[Attr.lidar_pos_x] ** 2 + df_data[Attr.lidar_pos_y] ** 2).astype(float))
    df_data[Attr.lidar_yaw] = df_data[Attr.lidar_yaw].apply(wrap_yaw).to_numpy()
    df_data = df_data[df_data["Distance"] <= 150]

    for category, data in df_data.groupby([Attr.category]):
        lidar_yaws = get_pos_angle(data[[Attr.lidar_pos_x, Attr.lidar_pos_y]].to_numpy())
        lidar_dist = data["Distance"].to_numpy()
        polar_histogram(lidar_yaws, lidar_dist, title=category, abin_num=16,
                        out_path=os.path.join(out_path, "position_distribution_{}.png".format(category)))
        histogram_2d_sns(data, title=category,
                         out_path=os.path.join(out_path, "lidar_points_{}.png".format(category)))

    title = "Overall"
    polar_histogram(get_pos_angle(df_data[[Attr.lidar_pos_x, Attr.lidar_pos_y]].to_numpy()),
                    df_data["Distance"],
                    title=title,
                    out_path=os.path.join(out_path, "position_distribution_{}.png".format(title)))
    histogram_2d_sns(df_data,
                     title=title,
                     out_path=os.path.join(out_path, "lidar_points_{}.png".format(title)))


if __name__ == "__main__":
    gt_data_path = "/mnt/data/autolabel/data/L2+_zdrive_urban"
    out_path = "/mnt/data/autolabel/data/L2+_zdrive_urban_statistic"
    os.makedirs(out_path, exist_ok=True)
    gt_obj = ObstacleClipGt(gt_data_path)

    histogram_on_each_category(gt_obj.data, out_path)
    num_histogram(gt_obj.data, out_path)
