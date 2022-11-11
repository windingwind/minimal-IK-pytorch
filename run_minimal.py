import numpy as np
import torch
from minimal.models import KinematicModel, KinematicPCAWrapper
from minimal.solver import Solver
from minimal.models_torch import KinematicModel as KinematicModelTorch, KinematicPCAWrapper as KinematicPCAWrapperTorch
from minimal.solver_torch import Solver as SolverTorch
import minimal.config as config


def single_minimal(jnts, pcl, save_path=None, gender="female", device="cpu", show_results=False):
    '''
    jnts: torch.Tensor | numpy.ndarray, N_joints*3(x, y, z)
    pcl: torch.Tensor | numpy.ndarray, N_points*3(x, y, z)
    save_path: str
    gender: 'male' | 'female' | 'neutral'
    device: torch.device
    show_results: show loss(mpl) and mesh(open3d) if True
    '''
    dbg_level = int(dbg_level)

    if device == "cpu":
        _device = None
        _Model = KinematicModel
        _Wrapper = KinematicPCAWrapper
        _Solver = Solver
    else:
        _device = torch.device(device)
        _Model = KinematicModelTorch
        _Wrapper = KinematicPCAWrapperTorch
        _Solver = SolverTorch

    smpl = _Model(device=_device).init_from_file(
        config.SMPL_MODEL_1_0_PATH[gender])

    wrapper = _Wrapper(smpl)
    solver = _Solver(wrapper)

    if dbg_level > -1:
        init_param = np.zeros(wrapper.n_params + 3)
        # translation
        init_param[:3] = -(jnts.max(axis=0) + jnts.min(axis=0))/2
        # rotation
        init_param[3] = np.pi/2

        solver.update_params(init_param)
        # o3d_plot([o3d_pcl(jnts, [0,0,1]), o3d_pcl(pcl, [1,0,0]), o3d_pcl(kpts_init, [0,1,0]), o3d_mesh(mesh_init)], 'Minimal Input')

    params_est, losses = solver.solve(
        jnts, pcl, show_results=show_results, mse_threshold=1e-4)
    print(losses[-1])

    mesh_est, keypoints_est = wrapper.run(params_est)
    # solver.save_model(os.path.join(save_path, ymdhms_time()+".obj"))
    if save_path is not None:
        solver.save_param(save_path)
    return [wrapper.core.verts.cpu().numpy(), wrapper.core.faces.cpu().numpy()], keypoints_est
