from extract import load_model
import matplotlib.pyplot as plt
from einops import rearrange
from collections import defaultdict
from tqdm import tqdm


def add_dataset(name, ax, skip=20, plot=True,  _x=None , **kwargs):
    print(f"Start: {name}")
    processor = load_model(name)
    dataset = processor.data_loader['extract'].dataset
    counts = set()

    index_bar = tqdm(range(0, len(dataset)), dynamic_ncols=True)
    for batch_idx in index_bar:
        (s, data, label, index) = dataset[batch_idx]
        for _label in label:
            counts.add(_label)

    #process = tqdm(dataset, dynamic_ncols=True)
    if ax is not None:
        index_bar = tqdm(range(0, len(dataset), skip), dynamic_ncols=True)
        for batch_idx in index_bar:
            (s, data, label, index) = dataset[batch_idx]
            mid_index = len(label) // 2
            _data = rearrange(data[mid_index], "c t v m -> t v (c m)")
            for x, y, z in _data[0]:
                if _x is not None:
                    x = _x
                ax.scatter(x, z, **kwargs)
        print(f"Done: {name}")
    return counts


fig, ax = plt.subplots()
#fig = ax = None

test_counts = add_dataset("/vol/research/extol/data/Phoenix2014T/Holistic/test/16.slt", ax, color="blue", _x=0.2)
dev_counts = add_dataset("/vol/research/extol/data/Phoenix2014T/Holistic/dev/16.slt", ax, color="red", _x=0.4)
train_counts = add_dataset("/vol/research/extol/data/Phoenix2014T/Holistic/train/16.slt", ax, color="green", _x=0.6)

combined = test_counts | train_counts
unique = dev_counts - combined
not_present = combined - dev_counts

print(f"dev - both: {len(unique)}/{len(dev_counts)}\nboth - dev: {len(not_present)}/{len(dev_counts)}")


if ax is not None:
    ax.set_aspect("equal")
    fig.savefig("/vol/research/SignRecognition/swisstxt/zzz2s.png")


"/vol/research/extol/data/Phoenix2014T/Holistic/dev/outputs.slt"