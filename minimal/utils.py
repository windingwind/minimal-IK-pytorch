import os
import time
import matplotlib.pyplot as plt
import numpy as np


SMPL_SKELETON_LINES = np.asarray([
    [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
    [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16],
    [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23],
    [22, 24], [23, 25], [10, 26], [11, 27], [15, 28]])


class WeightLoss(list):
    def __init__(self, weight=1) -> None:
        super().__init__()
        self.weight = weight

    def delta(self, idx=-1, absolute=True):
        if absolute:
            return abs(self[idx-1] - self[idx])
        else:
            return self[idx-1] - self[idx]

    def __getitem__(self, i):
        try:
            ret = super(WeightLoss, self).__getitem__(i) * self.weight
        except:
            ret = 0.
        return ret


class LossManager():
    def __init__(self, losses_with_weights={}, mse_threshold=1e-8, loss_threshold=1e-8) -> None:

        self.losses = {}
        self.epoch = 0
        self.mse_threshold = mse_threshold
        self.loss_threshold = loss_threshold
        self.add_loss(losses_with_weights)

        self.fig = None
        self.mpl_flag = True
        self.draw_flag = True

        self.arts = []

    def add_loss(self, losses_with_weights: dict):
        for loss_name, weight in losses_with_weights.items():
            self.losses[loss_name] = WeightLoss(weight)

    def update_loss(self, loss_name: str, value: float):
        if loss_name in self.losses.keys():
            self.losses[loss_name].append(float(value))

    def update_losses(self, losses_value: dict):
        for loss_name, value in losses_value.items():
            self.update_loss(loss_name, value)
        self.epoch += 1

    def save_losses(self, filepath: str):
        _losses = {}
        for loss_name, value in self.losses.items():
            _losses[loss_name] = np.array(value)
        np.savez(filepath, **_losses)

    def delta(self, idx=-1, absolute=True):
        return sum([loss.delta(idx, absolute) for loss in self.losses.values()])

    def check_losses(self):
        return self[-1] < self.loss_threshold or self.delta() < self.mse_threshold

    def str_losses(self, idx=-1):
        return "\t".join(["{}={:.4f}".format(loss_name, loss[idx]) for loss_name, loss in self.losses.items()])

    def show_losses(self):
        if self.mpl_flag:
            plt.ion()
            self.fig = plt.figure()
            self.fig.set_size_inches(12, 6)
            self.mpl_flag = False

        ax = self.fig.add_subplot(1, 2, 1)
        for loss_name, loss in self.losses.items():
            ax.plot(loss, label=loss_name)
        if "loss" not in self.arts:
            self.arts.append("loss")
            ax.legend()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs iterations")

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)

    def __len__(self):
        return min([len(loss) for loss in self.losses.values()])

    def __getitem__(self, i):
        return sum([loss[i] for loss in self.losses.values()])


def ymdhms_time(t: float = None) -> str:
    """
    Return current time str for print use

    :return: time str
    """
    if t is None:
        t = time.time()
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(t))


def file_paths_from_dir(path, extension='*', enable_print=True, collect_dir=False) -> list:
    """
    return specific files of the path and its sub-paths
    """
    if enable_print:
        print('Loading {} from {}'.format(extension, path))
    try:
        dir_list = os.listdir(path)
    except:
        dir_list = []
    filepaths = []
    for f in dir_list:
        # ignore hidden files
        if f.split('/')[-1][0] == '.':
            continue
        _path = os.path.join(path, f)
        if os.path.isdir(_path):
            if collect_dir:
                filepaths.append(_path)
            filepaths.extend(file_paths_from_dir(
                _path, extension, enable_print, collect_dir))

        elif extension != '!' and (extension == '*' or os.path.splitext(f)[1] == extension):
            filepaths.append(_path)
    return filepaths


def filename_decoder(file_path) -> dict:
    dir, _ = os.path.split(file_path)
    filename, extension = os.path.splitext(_)
    if '=' in filename:
        params = [param.split('=') for param in filename.split('_')]
    else:
        params = [[filename, " "]]
    params.append(["filepath", os.path.abspath(file_path)])
    return dict(params)
