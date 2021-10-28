import lzma
import pickle
import time
import random
import os
import yaml
from main import Processor, init_seed
from feeders.specs import holistic_joints, POINTS, left_hand_group, right_hand_group, upper_body_group, head_group, get_indexes
from feeders.feeder import process_holistic
import argparse
import torch
from einops import rearrange
from tqdm import tqdm
import copy
from pathlib import Path

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load(path):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def main(params):
    #  Load
    image_paths = {
        image_path.stem: image_path
        for image_path in sorted(Path(params.input_dir).rglob(f"*.{params.ext}"))
    }
    print(len(image_paths))
    dataset = [load(path) for stem, path in image_paths.items()]
    with open(f"{params.output_file}", "wb") as f:
        f.write(pickle.dumps(dataset))



if __name__ == '__main__':
    #  Get params
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/test", help="")
    parser.add_argument("--output_file", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/test/window_gcn.slt", help="")
    parser.add_argument("--ext", type=str,
                        default="feature", help="")
    params, _ = parser.parse_known_args()
    _time = time.time()
    main(params)
    print(f"Done in {time.time() - _time}s")