import pickle
import torch
from pytorch3d.structures import Meshes
from minimal.armatures import SMPLArmature


class KinematicModel(torch.nn.Module):
    """
    Kinematic model that takes in model parameters and outputs mesh, keypoints,
    etc.
    """

    def __init__(self, device=torch.device("cpu")):
        torch.backends.cudnn.benchmark = True
        super().__init__()
        self.device = device

    def init_from_model(self, model):
        for name in ["pose_pca_basis", "pose_pca_mean", "J_regressor", "skinning_weights",
                     "mesh_shape_basis", "mesh_template", "faces", "parents", "scale",
                     "n_shape_params", "n_joints", "compute_mesh", "J_regressor_ext"]:
            setattr(self, name, getattr(model, name))

        for name in ["coord_origin"]:
            _t = getattr(model, name)
            setattr(self, name, _t.clone())

        self.pose = torch.zeros((self.n_joints, 3), device=self.device)
        self.shape = torch.zeros(self.n_shape_params, device=self.device)

        self.verts = None
        self.mesh = None
        self.J = None
        self.R = None
        self.keypoints = None

        self.J_regressor_ext = model.J_regressor_ext

        return self

    def init_from_file(self, model_path, armature=SMPLArmature(), scale=1, compute_mesh=True):
        """
        Parameters
        ----------
        model_path : str
          Path to the model to be loaded.
        armature : object
          An armature class from `armatures.py`.
        scale : int, optional
          Scale of the model to make the solving easier, by default 1
        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

            self.pose_pca_basis = torch.from_numpy(
                params['pose_pca_basis']).to(self.device)
            self.pose_pca_mean = torch.from_numpy(
                params['pose_pca_mean']).to(self.device)

            self.J_regressor = torch.from_numpy(
                params['J_regressor']).to(self.device)

            self.skinning_weights = torch.from_numpy(
                params['skinning_weights']).to(self.device)

            self.mesh_pose_basis = torch.from_numpy(
                params['mesh_pose_basis']).to(self.device)  # pose blend shape
            self.mesh_shape_basis = torch.from_numpy(
                params['mesh_shape_basis']).to(self.device)
            self.mesh_template = torch.from_numpy(
                params['mesh_template']).to(self.device)

            self.faces = torch.from_numpy(
                params['faces'].astype(int)).to(self.device)

            self.parents = params['parents']

        self.n_shape_params = self.mesh_shape_basis.shape[-1]
        self.coord_origin = torch.tensor([0, 0, 0], device=self.device)

        self.scale = torch.tensor(scale).to(self.device)
        self.compute_mesh = compute_mesh
        self.mesh = None

        self.armature = armature
        self.n_joints = self.armature.n_joints
        self.pose = torch.zeros((self.n_joints, 3), device=self.device)
        self.shape = torch.zeros(self.n_shape_params, device=self.device)
        self.verts = None
        self.J = None
        self.R = None
        self.keypoints = None

        self.J_regressor_ext = \
            torch.zeros([self.armature.n_keypoints, self.J_regressor.shape[1]]).to(
                self.device)
        self.J_regressor_ext[:self.armature.n_joints] = self.J_regressor
        for i, v in enumerate(self.armature.keypoints_ext):
            self.J_regressor_ext[i + self.armature.n_joints, v] = 1
        self.J_regressor_ext = self.J_regressor_ext.to(
            device=self.device, dtype=torch.float64)

        return self

    def set_params(self, coord_origin: torch.Tensor = None, pose_abs: torch.Tensor = None, pose_pca: torch.Tensor = None, pose_glb: torch.Tensor = None, shape: torch.Tensor = None):
        """
        Set model parameters and get the mesh. Do not set `pose_abs` and `pose_pca`
        at the same time.

        Parameters
        ----------

        pose_abs : np.ndarray---> tensor, shape [n_joints, 3], optional
          The absolute model pose in axis-angle, by default None
        pose_pca : np.ndarray, optional
          The PCA coefficients of the pose, shape [n_pose, 3], by default None
        pose_glb : np.ndarray, shape [1, 3], optional
          Global rotation for the model, by default None
        shape : np.ndarray, shape [n_shape], optional
          Shape coefficients of the pose, by default None

        Returns
        -------
        np.ndarray, shape [N, 3]
          Vertices coordinates of the mesh, scale applied.
        np.ndarray, shape [K, 3]
          Keypoints coordinates of the model, scale applied.
        """
        if coord_origin is not None:
            self.coord_origin = coord_origin.clone()
        if pose_abs is not None:
            self.pose = pose_abs.clone()
        elif pose_pca is not None:
            pose_pca = pose_pca.clone()
            self.pose = torch.matmul(
                pose_pca.unsqueeze(0), self.pose_pca_basis[:pose_pca.shape[0]]
            )[0] + self.pose_pca_mean
            self.pose = torch.reshape(self.pose, [self.n_joints - 1, 3])
            if pose_glb is None:
                pose_glb = torch.zeros([1, 3]).to(self.device)
            pose_glb = torch.reshape(pose_glb, [1, 3])
            self.pose = torch.cat([pose_glb, self.pose])
        if shape is not None:
            self.shape = shape
        return self.update()

    def update(self):
        """
        Re-compute vertices and keypoints with given parameters.

        Returns
        -------
        np.ndarray--->Tensor, shape [N, 3]
          Vertices coordinates of the mesh, scale applied.
        np.ndarray, shape [K, 3]
          Keypoints coordinates of the model, scale applied.
        """
        verts = self.mesh_template + \
            torch.matmul(self.mesh_shape_basis, self.shape)
        self.J = torch.matmul(self.J_regressor, verts)
        self.R = self.rodrigues(self.pose.view((-1, 1, 3)))
        G = []
        G.append(self.with_zeros(torch.hstack(
            (self.R[0], self.J[0, :].view([3, 1])))))

        for i in range(1, self.n_joints):
            G.append(torch.matmul(G[self.parents[i]], self.with_zeros(
                torch.cat([
                    self.R[i],
                    (self.J[i, :] - self.J[self.parents[i], :]).view([3, 1])
                ], dim=1)
            )))
        G = torch.stack(G, dim=0)

        G = G - self.pack(torch.matmul(
            G,
            torch.hstack([self.J, torch.zeros(
                [self.n_joints, 1], device=self.device)])
            .reshape([self.n_joints, 4, 1])
        ))
        T = torch.tensordot(self.skinning_weights, G, dims=[[1], [0]])
        verts = torch.hstack(
            (verts, torch.ones([verts.shape[0], 1], device=self.device)))

        self.verts = \
            torch.matmul(T, verts.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        # update verts
        self.verts *= self.scale
        self.verts -= self.coord_origin

        # update mesh
        if self.compute_mesh:
            if self.mesh is None:
                self.mesh = Meshes(verts=[self.verts.to(
                    torch.float32)], faces=[self.faces])
            else:
                self.mesh._verts_list = [self.verts.to(torch.float32)]
                self.mesh._compute_packed(True)

        self.keypoints = torch.matmul(self.J_regressor_ext, self.verts)

        return self.mesh, self.keypoints

    def rodrigues(self, r: torch.Tensor):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].
        """
        #r = r.to(self.device)
        eps = r.clone().normal_(std=1e-8)
        # dim cannot be tuple
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3).unsqueeze(dim=0)
                  + torch.zeros((theta_dim, 3, 3))).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    def with_zeros(self, x: torch.Tensor):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=x.device)
        ret = torch.cat((x, ones), dim=0)
        return ret

    def pack(self, x: torch.Tensor):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros((x.shape[0], 4, 3)).to(x.device)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def save_obj(self, path):
        """
        Save the SMPL model into .obj file.
        Parameter:
        ---------
        path: Path to save.
        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class KinematicPCAWrapper():
    """
    A wrapper for `KinematicsModel` to be compatible to the solver.
    """

    def __init__(self, core: KinematicModel):
        """
        Parameters
        ----------
        core : KinematicModel
          Core model to be manipulated.
        """
        self.core = core
        self.n_pose = (core.n_joints - 1) * 3
        self.n_shape = core.n_shape_params
        self.n_glb = 3
        self.n_coord = 3
        self.n_params = self.n_pose + self.n_shape + self.n_glb

    def run(self, params):
        """
        Set the parameters, return the corresponding result.

        Parameters
        ----------
        params : np.ndarray---> Tensor
          Model parameters.

        Returns
        -------
        np.ndarray
          Corresponding result.
        """
        shape, pose_pca, pose_glb, coord_origin = self.decode(params)
        return \
            self.core.set_params(
                coord_origin=coord_origin, pose_glb=pose_glb, pose_pca=pose_pca, shape=shape)

    def decode(self, params):
        """
        Decode the compact model parameters into semantic parameters.

        Parameters
        ----------
        params : np.ndarray--->Tensor
          Model parameters.

        Returns
        -------
        np.ndarray
          Shape parameters.
        np.ndarray
          Pose parameters.
        np.ndarray
          Global rotation.
        """
        params = params.to(self.core.device)
        coord_origin = params[:self.n_coord]
        pose_glb = params[self.n_coord:self.n_coord + self.n_glb]
        pose_pca = params[self.n_coord + self.n_glb:-self.n_shape]
        shape = params[-self.n_shape:]
        return shape, pose_pca, pose_glb, coord_origin
