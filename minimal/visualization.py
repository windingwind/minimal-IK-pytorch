import time
import os
from typing import Callable, Generator, Iterable, Union, overload
import numpy as np
import open3d as o3d
import torch


def o3d_plot(o3d_items: list, title="", show_coord=True, **kwargs):
    if show_coord:
        _items = o3d_items + [o3d_coord(**kwargs)]
    else:
        _items = o3d_items
    view = o3d.visualization.VisualizerWithKeyCallback()
    view.create_window()
    # render = view.get_render_option()
    # render.point_size = 0.5
    for item in _items:
        view.add_geometry(item)
    view.run()
    # o3d.visualization.draw_geometries(_items, title)


def o3d_coord(size=0.1, origin=[0, 0, 0]):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def o3d_pcl(pcl: np.ndarray = None, color: list = None, colors: list = None, last_update=None):
    _pcl = last_update
    if _pcl is None:
        _pcl = o3d.geometry.PointCloud()

    if pcl is not None and pcl.size != 0:
        if pcl.shape[0] > 1000000:
            # auto downsample
            pcl = pcl[np.random.choice(
                np.arange(pcl.shape[0]), size=1000000, replace=False)]
        _pcl.points = o3d.utility.Vector3dVector(pcl)
        if color is not None:
            _pcl.paint_uniform_color(color)
        if colors is not None:
            _pcl.colors = o3d.utility.Vector3dVector(colors)
    return _pcl


def o3d_box(upper_bounds: np.ndarray = None, lower_bounds: np.ndarray = None, color: list = [1, 0, 0], last_update=None):
    _box = last_update
    if _box is None:
        _box = o3d.geometry.LineSet()
        _box.lines = o3d.utility.Vector2iVector(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7]
            ]
        )

    if upper_bounds is not None and lower_bounds is not None:
        x_u, y_u, z_u = upper_bounds
        x_l, y_l, z_l = lower_bounds
        _box.points = o3d.utility.Vector3dVector(
            [
                [x_u, y_u, z_u],
                [x_l, y_u, z_u],
                [x_l, y_l, z_u],
                [x_u, y_l, z_u],
                [x_u, y_u, z_l],
                [x_l, y_u, z_l],
                [x_l, y_l, z_l],
                [x_u, y_l, z_l],
            ]
        )
    if color is None:
        colors = np.repeat([color], 8, axis=0)
        _box.colors = o3d.utility.Vector3dVector(colors)
    return _box


def o3d_lines(skeleton: np.ndarray = None, lines: np.ndarray = None,
              color: list = [1, 0, 0], colors: list = None,
              last_update=None):
    _lines = last_update
    if _lines is None:
        _lines = o3d.geometry.LineSet()

    if skeleton is not None:
        _lines.points = o3d.utility.Vector3dVector(skeleton)
    if lines is not None:
        _lines.lines = o3d.utility.Vector2iVector(lines)
        if colors is None:
            colors = np.repeat([color], lines.shape[0], axis=0)
        _lines.colors = o3d.utility.Vector3dVector(colors)
    return _lines


# @overload
# def o3d_mesh(mesh: Meshes, color: list, last_update: None): ...


@overload
def o3d_mesh(mesh: Iterable, color: list, last_update: None): ...


def o3d_mesh(mesh: Union[Iterable, o3d.geometry.TriangleMesh] = None, color: list = None,
             last_update=None):
    _mesh = last_update
    if _mesh is None:
        _mesh = o3d.geometry.TriangleMesh()

    if mesh is not None:
        # if isinstance(mesh, Meshes):
        #     _mesh.vertices = o3d.utility.Vector3dVector(mesh.verts_packed().cpu())
        #     _mesh.triangles = o3d.utility.Vector3iVector(mesh.faces_packed().cpu())
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            _mesh.vertices = mesh.vertices
            _mesh.triangles = mesh.triangles
        else:
            _mesh.vertices = o3d.utility.Vector3dVector(mesh[0])
            if mesh[1] is not None:
                _mesh.triangles = o3d.utility.Vector3iVector(mesh[1])
        if color is not None:
            _mesh.paint_uniform_color(color)
        _mesh.compute_vertex_normals()
    return _mesh


def o3d_smpl_mesh(params: Union[np.ndarray, np.lib.npyio.NpzFile, dict] = None, color: list = None,
                  model=None, device="cpu", last_update=None):
    from minimal.models import KinematicModel, KinematicPCAWrapper
    from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
    import minimal.config as config
    import minimal.armatures as armatures

    _mesh = last_update
    if _mesh is None:
        _mesh = o3d.geometry.TriangleMesh()

    if params is None:
        return o3d_mesh(None, color, _mesh)

    if model is None:
        if device == "cpu":
            model = KinematicPCAWrapper(KinematicModel().init_from_file(
                config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, compute_mesh=False))
        else:
            model = KinematicPCAWrapperTorch(KinematicModelTorch(torch.device(device)).init_from_file(
                config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature, compute_mesh=False))

    assert isinstance(model, (KinematicPCAWrapper,
                      KinematicPCAWrapperTorch)), "Undefined model"

    # Unzip dict params
    if isinstance(params, np.lib.npyio.NpzFile) or (isinstance(params, dict) and isinstance(params.values()[0], np.ndarray)):
        # Numpy dict input
        pose_params = params["pose"]
        shape_params = params["shape"]
        params = np.hstack([pose_params, shape_params])

    elif isinstance(params, dict) and isinstance(params.values()[0], torch.Tensor):
        # Torch Tensor dict input
        pose_params = params["pose"]
        shape_params = params["shape"]
        params = torch.cat([pose_params, shape_params]).to(torch.float64)

    # Convert unmatched data types
    if isinstance(model, KinematicPCAWrapper) and isinstance(params, torch.Tensor):
        params = params.cpu().numpy()
    elif isinstance(model, KinematicModelTorch) and isinstance(params, np.ndarray):
        params = torch.from_numpy(params).to(
            device=model.device, dtype=torch.float64)

    model.run(params)

    if isinstance(model, KinematicPCAWrapper):
        mesh = [model.core.verts, model.core.faces]
    else:
        mesh = [model.core.verts.cpu().numpy(), model.core.faces.cpu().numpy()]

    return o3d_mesh(mesh, color, last_update=_mesh)


def pcl_filter(pcl_box, pcl_target, bound=0.2, offset=0):
    """
    Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
    """
    upper_bound = pcl_box[:, :3].max(axis=0) + bound
    lower_bound = pcl_box[:, :3].min(axis=0) - bound
    lower_bound[2] += offset

    mask_x = (pcl_target[:, 0] >= lower_bound[0]) & (
        pcl_target[:, 0] <= upper_bound[0])
    mask_y = (pcl_target[:, 1] >= lower_bound[1]) & (
        pcl_target[:, 1] <= upper_bound[1])
    mask_z = (pcl_target[:, 2] >= lower_bound[2]) & (
        pcl_target[:, 2] <= upper_bound[2])
    index = mask_x & mask_y & mask_z
    return pcl_target[index]


class O3DItemUpdater():
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.update_item = func()

    def update(self, params: dict):
        self.func(last_update=self.update_item, **params)

    def get_item(self):
        return self.update_item


class O3DStreamPlot():
    pause = False
    speed_rate = 1

    def __init__(self, width=1600, height=1200, with_coord=True) -> None:
        self.view = o3d.visualization.VisualizerWithKeyCallback()
        self.view.create_window(width=width, height=height)
        self.ctr = self.view.get_view_control()
        self.render = self.view.get_render_option()
        try:
            self.render.point_size = 3.0
        except:
            print('No render setting')

        self.with_coord = with_coord
        self.first_render = True

        self.plot_funcs = dict()
        self.updater_dict = dict()
        self.init_updater()
        self.init_plot()
        self.init_key_cbk()

    def init_updater(self):
        self.plot_funcs = dict(
            exampel_pcl=o3d_pcl, example_skeleton=o3d_lines, example_mesh=o3d_mesh)
        raise RuntimeError(
            "'O3DStreamPlot.init_updater' method should be overriden")

    def init_plot(self):
        for updater_key, func in self.plot_funcs.items():
            updater = O3DItemUpdater(func)
            self.view.add_geometry(updater.get_item())
            if self.with_coord:
                self.view.add_geometry(o3d_coord())
            self.updater_dict[updater_key] = updater

    def init_key_cbk(self):
        key_map = dict(
            w=87, a=65, s=83, d=68, h=72, l=76, space=32, one=49, two=50, four=52
        )
        key_cbk = dict(
            w=lambda v: v.get_view_control().rotate(0, 40),
            a=lambda v: v.get_view_control().rotate(40, 0),
            s=lambda v: v.get_view_control().rotate(0, -40),
            d=lambda v: v.get_view_control().rotate(-40, 0),
            h=lambda v: v.get_view_control().scale(-2),
            l=lambda v: v.get_view_control().scale(2),
            space=lambda v: exec(
                "O3DStreamPlot.pause = not O3DStreamPlot.pause"),
            one=lambda v: exec("O3DStreamPlot.speed_rate = 1"),
            two=lambda v: exec("O3DStreamPlot.speed_rate = 2"),
            four=lambda v: exec("O3DStreamPlot.speed_rate = 4"),
        )

        for key, value in key_map.items():
            self.view.register_key_callback(value, key_cbk[key])

    def init_show(self):
        self.view.reset_view_point(True)
        self.first_render = False

    def update_plot(self):
        self.view.update_geometry(None)
        if self.first_render:
            self.init_show()
        self.view.poll_events()
        self.view.update_renderer()

    def show(self, gen: Generator = None, fps: float = 30, save_path: str = ''):
        # print("[O3DStreamPlot] rotate: W(left)/A(up)/S(down)/D(right); resize: L(-)/H(+); pause/resume: space; speed: 1(1x)/2(2x)/4(4x)")

        if gen is None:
            gen = self.generator()
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        tick = time.time()
        frame_idx = 0
        while True:
            duration = 1/(fps*self.speed_rate)
            while time.time() - tick < duration:
                continue

            # print("[O3DStreamPlot] {} FPS".format(1/(time.time() - tick)))

            tick = time.time()

            try:
                if not self.pause:
                    update_dict = next(gen)
            except StopIteration as e:
                break

            for updater_key, update_params in update_dict.items():
                if updater_key not in self.updater_dict.keys():
                    continue
                self.updater_dict[updater_key].update(update_params)
            self.update_plot()
            if save_path:
                self.view.capture_screen_image(os.path.join(
                    save_path, '{}.png'.format(frame_idx)), True)
            frame_idx += 1

        # self.close_view()

    def show_manual(self, update_dict):
        for updater_key, update_params in update_dict.items():
            if updater_key not in self.updater_dict.keys():
                continue
            self.updater_dict[updater_key].update(update_params)
        self.update_plot()

    def close_view(self):
        self.view.close()
        self.view.destroy_window()

    def generator(self):
        raise RuntimeError(
            "'O3DStreamPlot.generator' method should be overriden")
