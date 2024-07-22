import argparse

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from plugin.data.dataset import HDMapNetSemanticDataset
from plugin.data.image import denormalize_img
from plugin.data.utils import get_proj_mat, perspective
from nuscenes import NuScenes
from PIL import Image


def vis_label(dataroot, version):
    data_conf = {
        'image_size': (256, 704),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 0.5],
        'thickness': 5,
        'angle_class': 36,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'final_dim': (256, 704),
    }

    color_map = np.random.randint(0, 256, (256, 3))
    color_map[0] = np.array([0, 0, 0])
    colors_plt = ['r', 'b', 'g']
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    # dataset = HDMapNetDataset(nusc, version=version, dataroot=dataroot, data_conf=data_conf, is_train=False)
    dataset = HDMapNetSemanticDataset(nusc, version, dataroot, data_conf, is_train=True)
    
    # gt_path = os.path.join(dataroot, 'samples', 'GT')
    # if not os.path.exists(gt_path):
    #     os.mkdir(gt_path)

    car_img = Image.open('icon/car.png')
    for idx in tqdm.tqdm(range(0, len(dataset), 10)):
        rec = dataset.samples[idx]
        # imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec)
        vectors = dataset.get_vectors(rec)

        imgs, trans, rots, intrins, post_trans, post_rots, \
        lidar_data, lidar_mask, car_trans, yaw_pitch_roll, \
        semantic_masks, instance_masks, direction_masks = dataset[idx]

        # lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])
        H, W = instance_masks.shape
        xcor = (lidar_data[:, 0] - data_conf['xbound'][0]) // data_conf['xbound'][2]
        ycor = (lidar_data[:, 1] - data_conf['ybound'][0]) // data_conf['ybound'][2]
        xcor = np.clip(xcor, 0, W - 1).astype(int)
        ycor = np.clip(ycor, 0, H - 1).astype(int)
        instance_masks[ycor, xcor] = 10

        plt.figure('semantic map', figsize=(12, 6))
        plt.imshow(instance_masks, cmap='OrRd')

        # base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]
        # base_path = os.path.join(gt_path, base_path)

        # if not os.path.exists(base_path):
        #     os.mkdir(base_path)

        plt.figure('road vector', figsize=(12, 6))
        plt.xlim(-30, 30)
        plt.ylim(15, -15)

        for vector in vectors:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])

        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        # map_path = os.path.join(base_path, 'MAP.png')
        # plt.savefig(map_path, bbox_inches='tight', dpi=400)
        # plt.show()
        # plt.close()

        for img, intrin, rot, tran, cam in zip(imgs, intrins, rots, trans, data_conf['cams']):
            img = denormalize_img(img)
            P = get_proj_mat(intrin, rot, tran)
            plt.figure(cam, figsize=(12, 6))
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.xlim(1600, 0)
            plt.ylim(900, 0)
            plt.axis('off')
            for vector in vectors:
                pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num]
                zeros = np.zeros((pts_num, 1))
                ones = np.ones((pts_num, 1))
                world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
                pix_coords = perspective(world_coords, P)
                x = np.array([pts[0] for pts in pix_coords])
                y = np.array([pts[1] for pts in pix_coords])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
                           angles='xy', scale=1, color=colors_plt[line_type])

            # cam_path = os.path.join(base_path, f'{cam}.png')
            # plt.savefig(cam_path, bbox_inches='tight', pad_inches=0, dpi=400)
            # plt.close()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local HD Map Demo.')
    parser.add_argument('dataroot', type=str, default='/mnt/SSD/ws/data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    args = parser.parse_args()

    vis_label(args.dataroot, args.version)
