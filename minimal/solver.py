from time import time
from typing import Union
import numpy as np
import torch
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
from pytorch3d.structures import Pointclouds
from minimal.models import KinematicPCAWrapper
from minimal.utils import LossManager
from minimal.config import VPOSER_DIR
from minimal.visualization import o3d_mesh, o3d_plot
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


class Solver:
    def __init__(self, model: KinematicPCAWrapper, eps=1e-5):
        """
        Parameters
        ----------
        eps : float, optional
        Epsilon for derivative computation, by default 1e-5
        max_iter : int, optional
        Max iterations, by default 30
        mse_threshold : float, optional
        Early top when mse change is smaller than this threshold, by default 1e-8
        verbose : bool, optional
        Print information in each iteration, by default False
        """
        self.model = model
        self.eps = eps

        # coord_origin + pose_params
        self.pose_params = np.zeros(
            self.model.n_pose + self.model.n_glb + self.model.n_coord)
        self.shape_params = np.zeros(self.model.n_shape)

        self.vp, _ = load_model(VPOSER_DIR, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
        self.vp = self.vp.to(self.model.core.device)

    def solve(self, jnts_target, pcls_target, solve_type="full", losses_with_weights=None, max_iter=30,
              kpts_threshold=0.04, mse_threshold=1e-7, loss_threshold=1e-6, u=1e-3, v=1.5, show_results=False):
        if solve_type == "full":
            params = self.params()
        elif solve_type == "pose":
            params = self.pose_params

        if losses_with_weights is None:
            losses_with_weights = dict(
                kpts_losses=1,
                # angle_losses=1,
                edge_losses=100,
                face_losses=100,
            )

        jacobian = np.zeros([jnts_target.size, params.shape[0]])

        pcls = Pointclouds(
            [torch.tensor(pcls_target, dtype=torch.float32, device=self.model.core.device)])

        # accelerate draw
        pcls_vis = pcls_target[np.random.choice(np.arange(
            pcls_target.shape[0]), size=min(1000, pcls_target.shape[0]), replace=False)]

        losses = LossManager(losses_with_weights, mse_threshold,
                             loss_threshold)

        for i in range(int(max_iter)):
            t = time()
            # update modle
            self.vpose_mapper()
            mesh_updated, jnts_updated = self.model.run(self.params())

            # compute keypoints loss
            loss_kpts, residual = jnts_distance(
                jnts_updated, jnts_target, activate_distance=kpts_threshold)
            losses.update_loss("kpts_losses", loss_kpts)

            # compute edge loss
            if losses_with_weights["edge_losses"] > 0:
                loss_edge = point_mesh_edge_distance(mesh_updated, pcls).cpu()
                losses.update_loss("edge_losses", loss_edge)

            # compute face loss
            if losses_with_weights["face_losses"] > 0:
                loss_face = point_mesh_face_distance(mesh_updated, pcls).cpu()
                losses.update_loss("face_losses", loss_face)

            # check loss
            if losses.check_losses():
                break

            for k in range(params.shape[0]):
                jacobian[:, k] = np.hstack([self.get_derivative(k)])

            jtj = np.matmul(jacobian.T, jacobian)
            jtj = jtj + u * np.eye(jtj.shape[0])

            delta = np.matmul(
                np.matmul(np.linalg.inv(jtj), jacobian.T), residual
            ).ravel()
            params -= delta

            update = losses.delta(absolute=False)

            if update > 0 and update > losses.delta(idx=-2, absolute=False):
                u /= v
            else:
                u *= v
            self.update_params(params)

            print(time()-t)

        if show_results:
            losses.show_losses()
            o3d_plot([o3d_mesh(mesh_updated)])

        return self.params(), losses

    def params(self):
        return np.hstack([self.pose_params, self.shape_params])

    def update_params(self, params: Union[np.lib.npyio.NpzFile, np.ndarray]):
        if isinstance(params, (np.lib.npyio.NpzFile, dict)):
            self.pose_params = params["pose"]
            self.shape_params = params["shape"]
        elif params.shape[0] == self.model.n_pose + self.model.n_coord + self.model.n_glb:
            self.pose_params = params
        elif params.shape[0] == self.model.n_params + self.model.n_coord:
            self.pose_params, self.shape_params = params[:self.model.n_pose +
                                                         self.model.n_coord + self.model.n_glb], params[-self.model.n_shape:]
        else:
            raise RuntimeError("Invalid params")

    def vpose_mapper(self):
        poseSMPL = torch.from_numpy(
            np.array([self.pose_params[6:69]])).type(torch.float).to(self.model.core.device)
        poZ = self.vp.encode(poseSMPL).mean
        self.pose_params[6:69] = self.vp.decode(
            poZ)['pose_body'].contiguous().reshape(poseSMPL.shape[1]).cpu().numpy()

    def get_derivative(self, n):
        """
        Compute the derivative by adding and subtracting epsilon

        Parameters
        ----------
        model : object
        Model wrapper to be manipulated.
        params : np.ndarray
        Current model parameters.
        n : int
        The index of parameter.

        Returns
        -------
        np.ndarray
        Derivative with respect to the n-th parameter.
        """
        params1 = np.array(self.params())
        params2 = np.array(self.params())

        params1[n] += self.eps
        params2[n] -= self.eps

        origin_compute_mesh = self.model.core.compute_mesh
        self.model.core.compute_mesh = False

        res1 = self.model.run(params1)[1]
        res2 = self.model.run(params2)[1]

        self.model.core.compute_mesh = origin_compute_mesh

        d = (res1 - res2) / (2 * self.eps)

        return d.ravel()

    def save_param(self, file_path):
        np.savez(file_path, pose=self.pose_params, shape=self.shape_params,
                 vertices=self.model.core.verts, keypoints=self.model.core.keypoints)

    def save_model(self, file_path):
        self.model.core.save_obj(file_path)


def jnts_distance(kpts_updated, kpts_target, activate_distance):
    d = (kpts_updated - kpts_target)
    _filter = np.linalg.norm(d, axis=1) < activate_distance
    _filter[[13, 14]] = np.linalg.norm(
        d[[13, 14]], axis=1) < 12 * activate_distance
    _filter[[16, 17]] = np.linalg.norm(
        d[[16, 17]], axis=1) < 6 * activate_distance
    _filter[[1, 2, 3, 26, 27]] = np.linalg.norm(
        d[[1, 2, 3, 26, 27]], axis=1) < 4 * activate_distance
    residual = np.where(np.repeat(_filter.reshape(
        _filter.shape[0], 1), 3, axis=1), 0, d).reshape(kpts_updated.size, 1)
    loss_jnts = np.mean(np.square(residual))
    return loss_jnts, residual
