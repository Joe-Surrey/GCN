import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
import itertools

# AUTSL requires your decryption key if you want to load videos

config = SignDatasetConfig(name="mediapipe", version="1.0.0", include_video=False, include_pose="holistic", fps=30)
autsl = tfds.load(name='autsl', builder_kwargs={
    "config": config,
})

print()
for datum in itertools.islice(autsl["train"], 0, 10):
    print(datum['id'].numpy().decode('utf-8'), datum['gloss_id'].numpy())
