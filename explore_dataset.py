import argparse
from pathlib import Path
import shutil
import sys
import copy
from extract import load_model


processor = load_model("/vol/research/extol/personal/cihan/data/PHOENIX2014T/phoenix14t.czech.dev",
                       weights="/vol/research/SignRecognition/MS-G3D/work_dirs/phoenix_train_z/weights/weights-23-34500.pt",
                       config="/vol/research/SignRecognition/MS-G3D/work_dirs/phoenix_train_z/config.yaml",
                       feeder="feeders.feeder.CzechFeeder", phase="test")

processor.start()











