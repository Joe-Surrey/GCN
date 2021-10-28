import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
from feeders import tools
from .augmentations import Augmentor
from collections import defaultdict
import tqdm


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, test=False, **kwargs):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def load_dataset_file(self, filename="/vol/research/SignTranslation/data/ChaLearn2021/train/ChaLearn2021.train.openpose.fp32.slt.full"):
        print(f"Loading {filename}")
        with open(filename, "rb") as f:
            try:
                return pickle.load(f)
            except ValueError:
                return pickle.load(f)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)


import pickle
import lzma
from .specs import openpose_joints, holistic_joints, POINTS
class ChaLearnFeeder(Feeder):

    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False, test=False, extract=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, type='openpose', augmentations=("feeders.augmentations.random_mirror",), **kwargs):
        self.augmentor = Augmentor(augmentations)
        self.extract = extract
        joints = openpose_joints if type == 'openpose' else holistic_joints
        self.indexes = [joints[point] for point in POINTS["SMPLH"]]
        self.test = test
        super(ChaLearnFeeder, self).__init__(data_path, label_path,
                 random_choose, random_shift, random_move,
                 window_size, normalization, debug, use_mmap)
        print(f"self.normalization: {self.normalization}")
        print(f"self.random_shift: {self.random_shift}")
        print(f"self.random_choose: {self.random_choose}")
        print(f"self.window_size: {self.window_size}")
        print(f"self.random_move: {self.random_move}")



    def load_data(self):
        tmp = self.load_dataset_file(self.data_path)
        self.data = []
        self.label = []
        self.sample_name = []
        if self.extract:
            self.all = tmp

        for s in tmp:
            self.label.append(int(s["gloss"]))  # TODO gloss or text
            self.sample_name.append((s["name"]))
            self.data.append(process_holistic(s, indexes=self.indexes))

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]




    def __getitem__(self, index):
        if self.extract:
            return self.all[index], self.data[index], self.label[index], index

        data_numpy, label, index = super(ChaLearnFeeder, self).__getitem__(index)
        if not self.test:
            data_numpy = self.augmentor(data_numpy)

        return data_numpy, label, index


import matplotlib.pyplot as plt


def vis(keypoints, name="test", edges=None, min_=None, max_=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, )

    for index, keypoint in enumerate(keypoints):
        x, y, z = keypoint
        if (min_ is None or index >= min_) and (max_ is None or index < max_):
            ax.scatter(x, y, color="blue", )
            ax.text(x, y, str(index), size=10, zorder=1)
    fig.savefig(f"/vol/research/SignRecognition/MS-G3D/imgs/test/{name}.png", format='png', bbox_inches='tight', pad_inches=0, dpi=1200)
    #if edges is not None:


def process_holistic(s, indexes, start=None, end=None, features=(0, 1, 2)):
    keypoints = pickle.loads(lzma.decompress(s['sign']))
    keypoints = keypoints[start:end]

    # Holistic
    keypoints = keypoints.reshape((-1, 543, 4)).astype('float32')[:, :, features]  # [0, 1, 3] for visibility

    # Openpose
    # keypoints = keypoints.reshape((-1, 135, 3)).astype('float32')
    # shoulder_means = (keypoints[:, 5:6, :2] + keypoints[:, 6:7, :2]) / 2
    # keypoints[:, :, :2] -= shoulder_means
    # TODO Reconcile

    #keypoints[:, :, 1] = - keypoints[:, :, 1]
    keypoints = rearrange(keypoints, "T V C -> C T V ()")
    # T V C -> C T V M
    keypoints = keypoints[:, :, indexes]
    return keypoints


def process_openpose(s, indexes, start=None, end=None, features=(0, 1, 2)):
    pass


class PhoenixFeeder(Feeder):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False, test=False, extract=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, type='openpose', start=None,
                 end=None, feature_indexes=(0, 1, 2), body_type="SMPLH", augmentations=("feeders.augmentations.random_mirror",), **kwargs):
        self.augmentor = Augmentor(augmentations)

        self.end=end
        self.start=start
        self.extract = extract
        self.feature_indexes=feature_indexes
        if self.extract:
            print("Extracting")
        self.test = test
        joints = openpose_joints if type == 'openpose' else holistic_joints
        self.indexes = [joints[point] for point in POINTS[body_type]]
        self.process = process_openpose if type == 'openpose' else process_holistic

        self.gloss_to_encoding = {}
        self.encoding_to_gloss = {}
        with open("/vol/research/SignRecognition/phoenix/stream-0.gloss-output.map", "r") as f:
            lines = f.read()
        for line in lines.split("\n"):
            if line != "":
                gloss, encoding = line.split(";")
                self.encoding_to_gloss[encoding] = gloss
                self.gloss_to_encoding[gloss] = encoding

        super(PhoenixFeeder, self).__init__(data_path, label_path,
                 random_choose, random_shift, random_move,
                 window_size, normalization, debug, use_mmap)
        print(f"self.normalization: {self.normalization}")
        print(f"self.random_shift: {self.random_shift}")
        print(f"self.random_choose: {self.random_choose}")
        print(f"self.window_size: {self.window_size}")
        print(f"self.random_move: {self.random_move}")

    def load_data(self):
        if not isinstance(self.data_path, list):
            self.data_path = [self.data_path]
        self.data = []
        self.label = []
        self.sample_name = []
        if self.extract:
            self.all = []
        for data_path in self.data_path:
            tmp = self.load_dataset_file(data_path)
            for s in tmp:
                alignments = s["alignments"]["pami0"].split(" ")
                sign = self.process(s, indexes=self.indexes, features=self.feature_indexes)
                chunk_length = 16
                # Split into chunks of 16
                if self.extract:
                    self.data.append([])
                    self.label.append([])
                for start in range((len(alignments) - chunk_length) + 1):
                    label = int(alignments[start + (chunk_length // 2)])
                    label = (label + 2) // 3

                    if self.extract:
                        self.data[-1].append(sign[:, start:start + chunk_length].copy())
                        self.label[-1].append(label)
                    else:
                        self.label.append(label)
                        self.sample_name.append((s["name"] + "_" + str(start)))
                        self.data.append(sign[:, start:start + chunk_length].copy())

                if self.extract:
                    self.all.append(s)
                    self.sample_name.append((s["name"]))

            #print(f"Largest sample: {max([sample.shape[1] for sample in self.data])}")
            if not self.extract:
                print(f"Largest class: {max(self.label)}")
            del tmp


        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        elif self.start is not None or self.end is not None:
            self.label = self.label[self.start:self.end]
            self.data = self.data[self.start:self.end]
            self.sample_name = self.sample_name[self.start:self.end]



    def __getitem__(self, index):
        if self.extract:
            return self.all[index], self.data[index], self.label[index], index

        data_numpy, label, index = super().__getitem__(index)
        if not self.test:
            data_numpy =  self.augmentor(data_numpy)

        return data_numpy, label, index

def unlistify(s):
    for key in ("name", "signer", "gloss", "text"):
        if isinstance(s[key], list):
            s[key] = s[key][0]
    for key in s['alignments']:
        if isinstance(s['alignments'][key], list):
            s['alignments'][key] = s['alignments'][key][0]


class FeatureFeeder(Feeder): # A feeder for extracted features
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, test=False, **kwargs):
        super(FeatureFeeder, self).__init__(data_path, label_path,
                                            random_choose, random_shift, random_move,
                                            window_size, normalization, debug, use_mmap)

    def load_data(self):
        tmp = self.load_dataset_file(self.data_path)
        self.data = []
        self.label = []
        self.sample_name = []
        chunk_length = 16
        for s in tmp:
            unlistify(s)
            whole_sign = pickle.loads(lzma.decompress(s['sign']))
            sign = [window for window in whole_sign]
            alignments = s["alignments"]["pami0"].split(" ")
            label = []
            for start in range((len(alignments) - chunk_length) + 1):
                window_label = int(alignments[start + (chunk_length // 2)])
                window_label = (window_label + 2) // 3
                label.append(window_label)
            assert len(label) == len(sign), f"Label: {len(label)}, sign: {len(sign)} -shape: ({whole_sign.shape})"
            self.label.extend(label)
            self.sample_name.extend([s["name"] for _ in range(len(sign))])
            self.data.extend(sign)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

"""            alignments = s["alignments"]["pami0"].split(" ")
            if any((item != "0" for item in alignments)):
                #starts = [index for index, encoding in enumerate(alignments) if self.encoding_to_gloss[encoding][-1] == "0"]
                #ends = [index for index, encoding in enumerate(alignments) if self.encoding_to_gloss[encoding][-1] == "2"]

                current_encoding = alignments[0]
                start = 0
                sign = process_holistic(s, indexes=self.indexes)
                for index, encoding in enumerate(alignments):
                    if encoding != current_encoding:
                        self.label.append(int(current_encoding))
                        self.sample_name.append((s["name"] + "_" + str(start)))
                        self.data.append(sign[:, start:index])
                        start = index
                        current_encoding = encoding
                self.label.append(int(current_encoding))
                self.sample_name.append((s["name"] + "_" + str(start)))
                self.data.append(sign[:, start:])
        print(f"Largest sample: {max([sample.shape[1] for sample in self.data])}")
        print(f"Largest class: {max(self.label)}")
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]"""