from extract import load_model
import matplotlib.pyplot as plt
from einops import rearrange
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from feeders.specs import name_neighbours, POINTS

# Test [-0.90234375, -0.89697266, -0.90234375]
# Dev [-0.96875   , -0.9355469 , -0.12164307]
# Train [-0.82958984, -0.89160156, -0.82958984]


def is_zero(item):
    # z_vec = [-0.90234375, -0.89697266, -0.90234375]  # Magic zero vector #  # [-0.82958984, -0.89160156, -0.82958984]
    return np.abs(item).sum() < 0.01


def add_dataset(name, skip=20, plot=True, **kwargs):
    print(f"Start: {name}")
    processor = load_model(name)
    dataset = processor.data_loader['extract'].dataset
    joints = POINTS["smplh"]
    neighbours = [(joints.index(beg), joints.index(end)) for beg, end in name_neighbours]
    index = 200

    (s, data, label, index) = dataset[index]
    for _t in range(len(data)):

        _data = rearrange(data[_t], "c t v m -> t v (c m)")

        num_zero = 0
        for t in range(len(_data)):
            fig, ax = plt.subplots()
            for coord in _data[t]:
                x, y, z = coord
                if not is_zero(coord):
                    ax.scatter(x, y, **kwargs)
                else:
                    ax.scatter(x, y, color="red")
                    num_zero += 1
            for beg, end in neighbours:
                beg_coord = _data[t, beg]
                end_coord = _data[t, end]

                if not is_zero(beg_coord) and not is_zero(end_coord):
                    xs = [beg_coord[0], end_coord[0]]
                    ys = [beg_coord[1], end_coord[1]]
                    ax.plot(xs, ys, **kwargs)
            ax.set_aspect("equal")
            plt.show()
            print()


def add_single(name, ax, **kwargs):
    print(f"Start: {name}")
    processor = load_model(name)
    dataset = processor.data_loader['extract'].dataset
    joints = POINTS["smplh"]
    neighbours = [(joints.index(beg), joints.index(end)) for beg, end in name_neighbours]
    index = 150

    (s, data, label, index) = dataset[index]
    _t = len(data) // 2

    _data = rearrange(data[_t], "c t v m -> t v (c m)")

    num_zero = 0
    t = _data.shape[0] // 2
    for coord in _data[t]:
        x, y, z = coord
        if not is_zero(coord):
            ax.scatter(x, y, **kwargs)
        else:
            ax.scatter(x, y, color="red")
            num_zero += 1

    for beg, end in neighbours:
        beg_coord = _data[t, beg]
        end_coord = _data[t, end]

        if not is_zero(beg_coord) and not is_zero(end_coord):
            xs = [beg_coord[0], end_coord[0]]
            ys = [beg_coord[1], end_coord[1]]
            ax.plot(xs, ys, **kwargs)



fig, ax = plt.subplots()
# fig = ax = None

add_single("/vol/research/extol/data/Phoenix2014T/Holistic/dev/16_zero.slt", ax, color="purple")

add_single("/vol/research/extol/data/Phoenix2014T/Holistic/train/16_zero.slt", ax, color="green")
add_single("/vol/research/extol/data/Phoenix2014T/Holistic/test/16_zero.slt", ax, color="blue")


ax.set_aspect("equal")
plt.show()
if ax is not None:
    ax.set_aspect("equal")
    fig.savefig("/vol/research/SignRecognition/swisstxt/comparesingle.png")


"/vol/research/extol/data/Phoenix2014T/Holistic/dev/outputs.slt"