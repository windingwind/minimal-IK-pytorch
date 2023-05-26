# Minimal IK Pytorch

A PyTorch implementation of [Minimal-IK](https://github.com/CalciferZh/Minimal-IK) to restore SMPL parameters from skeleton/keypoints + point cloud, with [VPoser](https://github.com/nghorbani/human_body_prior) pose prior for better performance.

> Note that this repository is separated from a larger project and is NOT fully tested.

Image from [Minimal-IK](https://github.com/CalciferZh/Minimal-IK). Just an example.
![image](https://user-images.githubusercontent.com/33902321/201305650-1ff6c93d-c2f2-4ddb-a85c-19ef14702b79.png)

## Installation

### Requirements
Test on Ubuntu 18.04, Python==3.8.8, torch==1.7.1+cu110, pytorch3d==0.4.0, human-body-prior==2.1.2.0
- PyTorch, NumPy
- pytorch3d(for loss computations): https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- human_body_prior(for SMPL pose prior): https://github.com/nghorbani/human_body_prior
- matplotlib, open3d(for visualization): `pip install matplotlib open3d`
- numba(for faster point cloud filter, not necessary)

### Pretrained Models
Please download [data.zip](https://zjueducn-my.sharepoint.com/:u:/g/personal/xy_wong_zju_edu_cn/EZSE66yWPgpFigG6nK9wM6QBx_PlXPwlzAy-xo5dSxvDXw) and unzip it under the current directory.

### Prepare Models
We process SMPL models following [Minimal-IK](https://github.com/CalciferZh/Minimal-IK). We provide models that are already processed and can be directly used.

## Data Structure

### SMPL Params
#### Pose
pose: 1 * 72(3 coord + (24 joints - 1 pelvis) * 3 pose)  
> pose range is [-1, 1]  

#### Shape
shape: 1 * 10  
> shape range is [0, 1]

## Usage
### Use Solver Directly

```python
# init solver
smpl = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

wrapper = KinematicPCAWrapper(smpl)
solver = Solver(wrapper)

# solve full: jnts(29 * 3), pcl(N * 3)
mesh, losses = solver.solve(jnts, pcl, "full")

# solve pose only
mesh, losses = solver.solve(jnts, pcl, "pose")

# get SMPL params: 1 * 82
params = solver.params()
```

### Initialize SMPL Parameters

```python
# params can be either np.lib.npyio.NpzFile(from np.load), ndarray(1 * 72), or ndarray(1 * 82)
solver.update_params(params)
mesh, jnts = solver.model.run(solver.params())
```

### Run Single Frame Optimization
```python
from run_minimal import single_minimal

(verts, faces), kpts = single_minimal(jnts, pcl, "/path/to/save", gender="female", device="cuda:0", show_results=True)
```

## Visualization
### Mesh
```python
import open3d as o3d
from minimal.visualization import o3d_plot, o3d_mesh

# mesh is from solver.solve, model.run, or obj file
mesh = o3d.io.read_triangle_mesh(os.path.join(obj_path, obj_name))
o3d_plot([o3d_mesh(mesh)])
```

# Bibliography
This repository is created along with the paper [mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar](https://dl.acm.org/doi/abs/10.1145/3503161.3548262)([Homepage](https://chen3110.github.io/mmbody)). This repo is a tool for early-stage test and might not be used in the final version of the paper.

Please consider citing our paper:
```
@inproceedings{chen2022mmbody,
  title={mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar},
  author={Chen, Anjun and Wang, Xiangyu and Zhu, Shaohao and Li, Yanxu and Chen, Jiming and Ye, Qi},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3501--3510},
  year={2022}
}
```
