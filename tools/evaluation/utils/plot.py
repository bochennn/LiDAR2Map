import pickle
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.font_manager as fm
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import open3d as o3d
import numpy as np


palette = {"gt": (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
           "manual_anno": (0.45, 0.45, 0.45),
           "detection_on_mc": (0.45, 0.45, 0.45),
           "detection": (1, 0, 0),
           "radar_3_frames": (1, 0, 0),
           "poly-mot": (0.45, 0.45, 0.45),
           "recognition": (0.984313725490196, 0.6039215686274509, 0.6),
           "radar_10_frames": (0.984313725490196, 0.6039215686274509, 0.6),
           "fusion": (0.2, 0.6274509803921569, 0.17254901960784313),
           "radar_front": (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
           "radar_front_raw": (1.0, 0.4980392156862745, 0.0),
           "radar_front_left": (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
           "radar_front_right": (0.792156862745098, 0.6980392156862745, 0.8392156862745098),
           "radar_rear": (1.0, 0.4980392156862745, 0.0),
           "radar_rear_raw": (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
           "radar_rear_left": (0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
           "radar_rear_right": (0.6941176470588235, 0.34901960784313724, 0.1568627450980392)}


def get_ax4(set_font=True):
    if set_font:
        sns.set_context("paper", rc={"font.size": 36,
                                     "axes.titlesize": 28,
                                     "axes.labelsize": 20,
                                     "xtick.labelsize": 20,
                                     "ytick.labelsize": 20,
                                     "xtick.titlesize": 20,
                                     "ytick.titlesize": 20,
                                     "legend.fontsize": 20,
                                     "legend.title_fontsize": 20,
                                     "lines.linewidth": 2})
    fig, ax_array = plt.subplots(2, 2)
    # fig.set_size_inches(60, 25)
    fig.set_size_inches(120, 50)
    # fig.set_dpi(150)
    fig.set_dpi(60)
    ax1 = ax_array[0][0]
    ax2 = ax_array[1][0]
    ax3 = ax_array[0][1]
    ax4 = ax_array[1][1]
    return fig, ax1, ax2, ax3, ax4


def get_ax2(set_font=True):
    if set_font:
        sns.set_context("paper", rc={"font.size": 48,
                                     "axes.titlesize": 36,
                                     "axes.labelsize": 30,
                                     "xtick.labelsize": 30,
                                     "ytick.labelsize": 30,
                                     "xtick.titlesize": 30,
                                     "ytick.titlesize": 30,
                                     "legend.fontsize": 30,
                                     "legend.title_fontsize": 30,
                                     "lines.linewidth": 5})
    fig, ax_array = plt.subplots(2, 1)
    fig.set_size_inches(70, 40)
    fig.set_dpi(80)
    ax1 = ax_array[0]
    ax2 = ax_array[1]
    return fig, ax1, ax2


def get_ax1():
    fig, ax = plt.subplots()
    fig.set_size_inches(70, 40)
    fig.set_dpi(80)
    return fig, ax


def plot_from_pickle(file_path):
    fig = pickle.load(open(file_path, 'rb'))
    plt.show()


def text_o3d(text, pos, direction=None, degree=-90, color=(255, 0, 0), font_size=100):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param color: color of text, red in default
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=color)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def polar_histogram_core_for_match(azimut_gt, radius_gt, azimut_fn, radius_fn, fig, ax, rbin_num=6, abin_num=9, title=None):
    rbins = np.linspace(0, 150, rbin_num)
    abins = np.linspace(0, 2 * np.pi, abin_num)
    subdivs = 150
    abins_sub = np.linspace(0, 2 * np.pi, (len(abins) - 1) * subdivs + 1)
    A, R = np.meshgrid(abins_sub, rbins)
    hist_gt, _, _ = np.histogram2d(azimut_gt, radius_gt, bins=(abins, rbins))
    hist_fn, _, _ = np.histogram2d(azimut_fn, radius_fn, bins=(abins, rbins))
    pc = ax.pcolormesh(A, R, np.repeat(hist_fn.T, subdivs, axis=1), cmap="Blues")
    text_abins = [(abins[i] + abins[i + 1]) / 2 for i in range(len(abins) - 1)]
    text_rbins = [(rbins[i] + rbins[i + 1]) / 2 for i in range(len(rbins) - 1)]
    for aidx, abin in enumerate(text_abins):
        for ridx, rbin in enumerate(text_rbins):
            gt_num = hist_gt[aidx][ridx]
            fn_num = hist_fn[aidx][ridx]
            text = "{}, {:.2f}%".format(int(fn_num), (fn_num/gt_num) * 100)
            ax.text(abin, rbin, text, fontsize=8, ha="center")
    # yticks = [50, 100, 150]
    yticks = [30, 60, 90, 120, 150]
    yticks_label = ["{}m".format(value) for value in yticks]
    ax.set_yticks(yticks, yticks_label)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    fig.colorbar(pc, pad=0.1, ax=ax)
    if title:
        ax.set_title(title, y=1.1, fontsize=13)

def polar_histogram_core(azimut, radius, fig, ax, rbin_num=15, abin_num=24, title=None):
    rbins = np.linspace(0, 150, rbin_num)
    abins = np.linspace(0, 2 * np.pi, abin_num)
    subdivs = 150
    abins_sub = np.linspace(0, 2 * np.pi, (len(abins) - 1) * subdivs + 1)
    hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
    A, R = np.meshgrid(abins_sub, rbins)
    pc = ax.pcolormesh(A, R, np.repeat(hist.T, subdivs, axis=1), cmap="Blues")

    text_abins = [(abins[i] + abins[i+1]) / 2 for i in range(len(abins) - 1)]
    text_rbins = [(rbins[i] + rbins[i+1]) / 2 for i in range(len(rbins) - 1)]
    for aidx, abin in enumerate(text_abins):
        for ridx, rbin in enumerate(text_rbins):
            num_str = str(int(hist[aidx][ridx]))
            ax.text(abin, rbin, num_str, fontsize=8, ha="center")
    # yticks = [50, 100, 150]
    yticks = [30, 60, 90, 120, 150]
    yticks_label = ["{}m".format(value) for value in yticks]
    ax.set_yticks(yticks, yticks_label)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    fig.colorbar(pc, pad=0.1, ax=ax)
    if title:
        ax.set_title(title, y=1.1, fontsize=13)
    # plt.tight_layout()


def polar_histogram(azimut, radius, rbin_num=15, abin_num=24, title=None, out_path=None):
    """
    Histogram of position distribution of object samples
    Args:
        azimut:
        radius:
        rbin_num:
        abin_num:
        title:
        out_path:

    Returns:


    """
    rbins = np.linspace(0, radius.max(), rbin_num)
    abins = np.linspace(0, 2 * np.pi, abin_num)
    subdivs = 150
    abins_sub = np.linspace(0, 2 * np.pi, (len(abins) - 1) * subdivs + 1)
    hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
    A, R = np.meshgrid(abins_sub, rbins)
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(projection="polar"))

    pc = ax.pcolormesh(A, R, np.repeat(hist.T, subdivs, axis=1), cmap="Blues")
    yticks = [50, 100, 150]
    yticks_label = ["{}m".format(value) for value in yticks]
    ax.set_yticks(yticks, yticks_label)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    fig.colorbar(pc, pad=0.1, ax=ax)
    if title:
        ax.set_title(title, y=1.1, fontsize=13)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def histogram_2d_sns(data, title=None, out_path=None):
    data = data[data["Lidar points in box"] > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.histplot(data, x="Distance", y="Lidar points in box",
                      bins=20, discrete=(False, False), log_scale=(False, True),
                      ax=ax, cmap="Blues", cbar=True)
    ax.set_aspect(30)
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    data_path = sys.argv[1]
    if data_path.endswith("pkl"):
        plot_from_pickle(data_path)
