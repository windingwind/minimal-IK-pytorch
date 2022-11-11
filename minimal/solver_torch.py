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
        self.device = model.core.device

        # coord_origin + pose_params
        self.pose_params = torch.zeros(
            self.model.n_pose + self.model.n_glb + self.model.n_coord, device=self.device)
        self.shape_params = torch.zeros(self.model.n_shape, device=self.device)

        self.vp, _ = load_model(VPOSER_DIR, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
        self.vp = self.vp.to(self.device)

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

        jacobian = torch.zeros(
            [jnts_target.size, params.shape[0]], device=self.device)

        jnts_target = torch.from_numpy(jnts_target).to(self.device)
        pcls = Pointclouds(
            [torch.tensor(pcls_target, dtype=torch.float32, device=self.device)])

        # accelerate draw
        pcls_vis = pcls_target[np.random.choice(np.arange(
            pcls_target.shape[0]), size=min(1000, pcls_target.shape[0]), replace=False)]

        losses = LossManager(losses_with_weights, mse_threshold,
                             loss_threshold)

        for i in range(int(max_iter)):
            t = time()
            # update model
            self.vpose_mapper()
            mesh_updated, jnts_updated = self.model.run(self.params())

            # compute keypoints loss
            loss_kpts, residual = jnts_distance(
                jnts_updated, jnts_target, activate_distance=kpts_threshold)
            losses.update_loss("kpts_losses", loss_kpts.cpu())

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
                jacobian[:, k] = self.get_derivative(k)

            jtj = torch.matmul(jacobian.T, jacobian)
            jtj = jtj + u * torch.eye(jtj.shape[0], device=self.device)

            delta = torch.matmul(
                torch.matmul(torch.inverse(jtj), jacobian.T), residual.to(
                    jacobian.dtype)
            ).view(-1)
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
        return torch.cat([self.pose_params, self.shape_params]).to(torch.float64)

    def update_params(self, params: Union[np.lib.npyio.NpzFile, np.ndarray, torch.Tensor]):
        if isinstance(params, np.lib.npyio.NpzFile):
            self.pose_params = torch.from_numpy(params["pose"]).to(self.device)
            self.shape_params = torch.from_numpy(
                params["shape"]).to(self.device)
            return

        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params).to(self.device)

        if params.shape[0] == self.model.n_pose + self.model.n_coord + self.model.n_glb:
            self.pose_params = params
        elif params.shape[0] == self.model.n_params + self.model.n_coord:
            self.pose_params, self.shape_params = params[:self.model.n_pose +
                                                         self.model.n_coord + self.model.n_glb], params[-self.model.n_shape:]
        else:
            raise RuntimeError("Invalid params")

    def vpose_mapper(self):
        poseSMPL = self.pose_params[6:69].to(torch.float32)
        poZ = self.vp.encode(poseSMPL.view(1, poseSMPL.shape[0])).mean
        self.pose_params[6:69] = self.vp.decode(
            poZ)['pose_body'].contiguous().reshape(poseSMPL.shape[0]).to(torch.float64)

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
        params1 = self.params()
        params2 = self.params()

        params1[n] += self.eps
        params2[n] -= self.eps

        origin_compute_mesh = self.model.core.compute_mesh
        self.model.core.compute_mesh = False

        res1 = self.model.run(params1)[1]
        res2 = self.model.run(params2)[1]

        self.model.core.compute_mesh = origin_compute_mesh

        d = (res1 - res2) / (2 * self.eps)

        return d.view(-1)

    def save_param(self, file_path):
        if isinstance(self.shape_params, torch.Tensor):
            shape = self.shape_params.cpu().numpy()
        else:
            shape = self.shape_params
        np.savez(file_path, pose=self.pose_params.cpu().numpy(), shape=shape, vertices=self.model.core.verts.cpu(
        ).numpy(), keypoints=self.model.core.keypoints.cpu().numpy())

    def save_model(self, file_path):
        self.model.core.save_obj(file_path)


def jnts_distance(kpts_updated: torch.Tensor, kpts_target: torch.Tensor, activate_distance):
    d = (kpts_updated - kpts_target)
    _filter = torch.norm(d, dim=1) < torch.tensor(
        activate_distance, device=d.device)
    _filter[[13, 14]] = torch.norm(d[[13, 14]], dim=1) < torch.tensor(
        12 * activate_distance, device=d.device)
    _filter[[16, 17]] = torch.norm(d[[16, 17]], dim=1) < torch.tensor(
        6 * activate_distance, device=d.device)
    _filter[[1, 2, 3, 26, 27]] = torch.norm(d[[1, 2, 3, 26, 27]], dim=1) < torch.tensor(
        4 * activate_distance, device=d.device)
    residual = torch.where(_filter.view(
        _filter.shape[0], 1).expand(d.shape), 0., d).to(d.device)
    loss_jnts = torch.mean(torch.square(residual)).to(d.device)
    return loss_jnts, residual.view(-1)
