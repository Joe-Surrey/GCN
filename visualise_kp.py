import pickle
import lzma
import numpy as np
import argparse
from einops import rearrange
import cv2
import matplotlib.pyplot as plt


EPSILON = 0.00001


def is_zero(features):
    return np.abs(features).sum(axis=-1) < 0.1


def normalise(features):
    # Normalise by shoulder centres and hip centres (ignore visibility)
    shoulder_centres = features[:, 11:13, :-1].mean(axis=1)[:, np.newaxis, :]
    # 11 and 12 are shoulders  https://raw.githubusercontent.com/google/mediapipe/b899d17f185f6bcbf3f5947d3e134f8ce1e69407/mediapipe/modules/pose_landmark/pose_landmark_topology.svg
    hip_centres = features[:, 23:25, :2].mean(axis=1)[:, np.newaxis, :]  # 23 and 24 are hips

    height = np.linalg.norm(shoulder_centres[:, :, :2] - hip_centres, axis=-1).mean()
    if height > 0.000001:
        zero_indexes = is_zero(features)
        # Centre on shoulders
        features[:, :, :-1] -= shoulder_centres

        # Scale by height
        features[:, :, :-1] /= height

        features[zero_indexes] = 0
       ## Scale z
       #z_min = features[:, :, 2].min(axis=1)
       #features[:, :, 2] -= z_min[:, np.newaxis]
       #z_max = features[:, :, 2].max(axis=1)
       #features[:, :, 2] /= z_max[:, np.newaxis]
        features += EPSILON
        return features
    print("Height = 0, removed")
    return None


def rescale(features):
    features[:, :, :2] *= 512
    return features


def verify(features):
    return np.isnan(features).any()

# Load file
def main(params):
    try:
        with open(params.file, "rb") as f:
            dataset = pickle.load(f)
    except Exception:
        print(f"File not found: {params.file}")
        exit(1)

    # Get means
    base_name = "/vol/research/extol/data/Phoenix2014T/Videos720x720px"

    for s in dataset:
        video_name = f"{base_name}/{s['name']}.mp4"
        cap = cv2.VideoCapture(video_name)
        length = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if not cap.isOpened():
            print(f"Error opening {video_name}")
            return
        features = rearrange(pickle.loads(lzma.decompress(s['sign'])), "T (V XYZV) -> T V XYZV", V=543)
        #figs = []
        #axs = []
        for frame_index, keypoints in zip(range(length), features):
            if frame_index > 10:
                break
            success, image = cap.read()
            if success:
                fig, ax = plt.subplots()
                #figs.append(fig)
                #axs.append(ax)
                ax.imshow(image)

                for _x, _y, _z, conf in keypoints:
                    x = 0 if (abs(_x) > 1 or abs(_y) > 1) else _x
                    y = 0 if (abs(_x) > 1 or abs(_y) > 1) else _y
                    # print(x, y)

                    ax.scatter(round(x * width), round(y * height), color="red")
        print(length)
        plt.show()
        # input()
        # print()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/dev/16.slt", help="")
    parser.add_argument("--output_file", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/train/16_zero.slt", help="")

    params, _ = parser.parse_known_args()
    main(params)


# Load a video and keypoints
# plot on top