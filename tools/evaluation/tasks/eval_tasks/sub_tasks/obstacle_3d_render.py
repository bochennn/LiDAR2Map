import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from ....utils.plot import get_ax1


class ObstacleO3dRender:
    def __init__(self, engine="cv", visible=False):
        self.visible = visible
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=visible)
        self.vis.get_render_option().point_size = 1
        self.engine = engine
        self.plt_ax = None

    def __enter__(self):
        self.get_plt_ax()
        return self

    def __exit__(self, *args, **kwargs):
        self.vis.destroy_window()
        self.fig_close()

    def fig_close(self):
        if self.plt_ax is not None:
            plt.close(self.plt_ax[0])

    def get_plt_ax(self):
        if self.plt_ax is None:
            self.plt_ax = get_ax1()
        return self.plt_ax

    def generate_o3d_image(self, geometry_patches, out_file_path, translate, text=""):
        self.vis.clear_geometries()
        for geometry in geometry_patches:
            self.vis.add_geometry(geometry)
        self.vis.get_view_control().set_lookat([1, 0, 0])
        self.vis.get_view_control().translate(translate[0]*2, translate[1]*2)
        dist = np.linalg.norm([translate[0], translate[1]])
        if dist < 30:
            zoom_ratio = 0.05
        else:
            zoom_ratio = 0.1
        self.vis.get_view_control().set_zoom(zoom_ratio)
        if self.visible:
            self.vis.run()
            return
        else:
            self.vis.poll_events()
            self.vis.update_renderer()
        image = self.vis.capture_screen_float_buffer(True)

        if self.engine == "cv":
            image = np.asarray(image)[..., ::-1]
            image *= 255
            dy = 30
            if text:
                for idx, line_text in enumerate(text.split('\n')):
                    image = cv2.putText(image, line_text, (10, 40 + dy * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(out_file_path, image)
        elif self.engine == "matplotlib":
            fig, ax = self.get_plt_ax()
            ax.set_axis_off()
            ax.imshow(np.asarray(image))
            ax.text(10, 50, text.ljust(30),
                    size=30,
                    bbox={'facecolor': 'grey', 'edgecolor': 'none', 'boxstyle': 'round', 'alpha': 0.5, 'pad': 1})
            fig.savefig(out_file_path, bbox_inches='tight')
            ax.clear()
