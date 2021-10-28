import torch
import pickle
import lzma


def load_dataset_file(filename):

    print(f"Loading {filename}")
    with open(filename, "rb") as f:
        try:
            return pickle.load(f)
        except ValueError:
            import pickle5
            return pickle5.load(f)
        except Exception:
            import gzip
            return pickle.loads(gzip.decompress(f.read()))


gt = torch.zeros((1201,), dtype=torch.float32)
preds = torch.zeros((1201,), dtype=torch.float32)

annotation_file = "/vol/research/extol/data/Phoenix2014T/Holistic/dev/outputs.slt"

tmp = load_dataset_file(annotation_file)
for s in tmp:
    for label in (((int(item) + 2) // 3) for item in s['alignments']['pami0'].split(" ")):
        gt[label] += 1

    for f in pickle.loads(lzma.decompress(s['sign'])).argmax(dim=-1)[0]:
        preds[f] += 1

print(f"Ground truth: {gt.topk(k=20)}")
print(f"Predictions: {preds.topk(k=20)}")




# "/vol/research/extol/data/Phoenix2014T/Holistic/dev/outputs.slt"
#
# one_hot = torch.zeros_like(example)
#
# return one_hot