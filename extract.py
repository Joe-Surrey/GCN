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


weights = {
    'chalearn': "/vol/research/SignRecognition/MS-G3D/work_dirs/eps/weights/weights-91-39949.pt",
    'phoenix': "/vol/research/SignRecognition/MS-G3D/work_dirs/phoenix_train_z/weights/weights-160-224960.pt"#"/vol/research/SignRecognition/MS-G3D/work_dirs/phoenix_1000/weights/weights-124-1396240.pt",# weights-51-574260.pt",
}


def get_parser(key="phoenix"):
    # parameter priority: command line > config file > default
    parser = {
        "work_dir": "/vol/research/SignRecognition/MS-G3D/work_dirs/extract",
        'model_saved_name': '',
        'config': '/vol/research/SignRecognition/MS-G3D/config/phoenix_holistic_eps_test.yaml',
        # 'path to the configuration file',
        'assume_yes': 'store_true',
        # 'Say yes to every prompt')
        'phase':'extract',
        # 'must be train or test',
        'save_score': False,
        # 'if ture, the classification score will be stored')
        'seed': random.randrange(200),
        # 'random seed',
        'log_interval': 100,
        # 'the interval for printing messages (#iteration)',
        'save_interval': 1,
        # 'the interval for storing models (#iteration)',
        'eval_interval': 1,
        # 'the interval for evaluating models (#iteration)',
        'eval_start':1,
        # 'The epoch number to start evaluating models',
        'print_log':True,
        # 'print logging or not',
        'show_topk':[1, 5],
        # 'which Top K accuracy will be shown')
        'feeder':'feeder.feeder',
        # 'data loader will be used',
        'num_worker':os.cpu_count(),
        # 'the number of worker for data loader',
        'train_feeder_args':dict(),
        # 'the arguments of data loader for training',
        'test_feeder_args':dict(),
        # 'the arguments of data loader for test')
        'model':None,
        # 'the model will be used',
        'model_args':dict(),
        # 'the arguments of model',
         'weights': weights[key],
         #'weights':"/vol/research/SignRecognition/MS-G3D/work_dirs/phoenix_1000/weights/weights-124-1396240.pt",# weights-51-574260.pt",
        # 'the weights for network initialization',
        'ignore_weights':[],
        # 'the name of weights which will be ignored in the initialization',
        'half': False,
        # 'Use half_precision (FP16) training',
        'amp_opt_level': 1,
        # 'NVIDIA Apex AMP optimization level')
        'base_lr': 0.01,
        # 'initial learning rate',
        'step': [20, 40, 60],
        # 'the epoch where optimizer reduce the learning rate',
        'device': 0,
        # 'the indexes of GPUs for training or testing',
        'optimizer':'SGD',
        # 'type of optimizer',
        'nesterov': False,
        # 'use nesterov or not',
        'batch_size': 32,
        # 'training batch size',
        'test_batch_size': 256,
        # 'test batch size',
        'forward_batch_size': 16,
        # 'Batch size during forward pass, must be factor of batch_size',
        'start_epoch': 0,
        # 'start training from which epoch',
        'num_epoch': 80,
        # 'stop training in which epoch',
        'weight_decay': 0.0005,
        # 'weight decay for optimizer',
        'optimizer_states': None,
        # 'path of previously saved optimizer states',
        'checkpoint': None,
        # 'path of previously saved training checkpoint',
        'debug': False,
        # 'Debug mode; default false')
    }
    return Struct(**parser)


def load_model(data_path, weights="/vol/research/extol/data/Phoenix2014T/Holistic/test/16_eps.slt", feeder=None,
               config='/vol/research/SignRecognition/MS-G3D/config/phoenix_holistic_z.yaml', phase="extract"):
    #os.chdir("/vol/research/SignRecognition/MS-G3D")
    p = get_parser()  #

    p.config = config
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)

        p.__dict__.update(default_arg)
    if feeder is not None:
        p.feeder = feeder
    p.weights = weights
    p.phase = phase
    p.test_feeder_args["data_path"] = data_path
    init_seed(p.seed)
    processor = Processor(p)
    return processor




def main(params):
    #  Load
    name = params.input_file.split("/")[-1].split(".")[0]
    #with open(f"{params.input_file}", "rb") as f:
    #    dataset = pickle.loads(f.read())

    left_hand = get_indexes(left_hand_group, holistic_joints)
    right_hand = get_indexes(right_hand_group, holistic_joints)
    head = get_indexes(head_group, holistic_joints)
    upper_body = get_indexes(upper_body_group, holistic_joints)

    _time = time.time()
    #  Load processor
    processor = load_model(params.input_file, weights=params.weights, config=params.config)
    model = processor.model
    model.eval()
    print(f"Processor done in {time.time() - _time}s")

    indexes = [holistic_joints[point] for point in POINTS["SMPLH"]]
    # Run
    #dataset = [_s for _s in dataset] # if not Path(f"{params.output_file}/{_s['name']}.logit").exists()]
    #print(len(dataset))
    dataset = processor.data_loader['extract']
    #process = tqdm(dataset, dynamic_ncols=True)
    chunk_length = 16

    process = tqdm(dataset, dynamic_ncols=True)
    acc = 0
    count = 0
    for batch_idx, (s, data, label, index) in enumerate(process):
        file_name = f"{params.output_file}/{s['name'][0]}.feature"
        #features = []
        #for _data, _label in zip(data, label):
        data = torch.stack(data).squeeze(dim=1).float().cuda(processor.output_device)
        output = model(data, extract=True)
        left_hand_features = output[:, :, :, left_hand].mean(dim=-1)
        right_hand_features = output[:, :, :, right_hand].mean(dim=-1)
        head_features = output[:, :, :, head].mean(dim=-1)
        upper_body_features = output[:, :, :, upper_body].mean(dim=-1)

        features = torch.cat([left_hand_features, right_hand_features, head_features, upper_body_features],
                                                           dim=1).cpu()
        # count += 1
        # with open("/vol/research/SignRecognition/swisstxt/extract.pkl", "wb") as f:
        #     f.write(pickle.dumps([_data, output, _label]))
        #
        # print(_label.item(), output.argmax(dim=-1).item(), _label == output.argmax(dim=-1).item())
        # acc += 1 if (output.argmax(dim=-1).item() == _label.item()) else 0
        # count += 1
        # features = torch.stack(features)
        output = rearrange(features, "b f t -> b t f").to("cpu")
        s["sign"] = lzma.compress(pickle.dumps(output))
        with open(file_name, "wb") as f:
             f.write(pickle.dumps(s))
    #print(f"Accuracy: {(acc * 100) / count}%")
    #for _s in process:
    #    file_name = f"{params.output_file}/{_s['name']}.logit"
    #    if not Path(file_name).exists():
    #        features = None
    #        alignments = _s["alignments"]["pami0"].split(" ")
    #        keypoints = rearrange(torch.tensor(process_holistic(_s, indexes)), "c t v m -> () c t v m")
    #        for start in range((keypoints.shape[2] - chunk_length) + 1):
    #            label = int(alignments[start + (chunk_length // 2)])
    #            label = (label + 2) // 3
    #            data = keypoints[:, :, start:start + chunk_length].to(model.fc.bias.device)
    #            output = model(data).unsqueeze(-1)#, extract=True)
    #            with open("/vol/research/SignRecognition/swisstxt/extract.pkl", "wb") as f:  # TODO remove
    #                f.write(pickle.dumps([data, output, label]))
    #            if features is None:
    #                features = output
    #            else:
    #                features = torch.cat([features, output], dim=-1)
                ## Combine means of hands, head and body
                #left_hand_features = output[:, :, :, left_hand].mean(dim=-1)
                #right_hand_features = output[:, :, :, right_hand].mean(dim=-1)
                #head_features = output[:, :, :, head].mean(dim=-1)
                #upper_body_features = output[:, :, :, upper_body].mean(dim=-1)
    ##
                #if features is None:
                #    features = torch.cat([left_hand_features, right_hand_features, head_features, upper_body_features],
                #                              dim=1).cpu()
                #else:
                #    features = torch.cat([features, torch.cat([left_hand_features, right_hand_features, head_features, upper_body_features],
                #                              dim=1).cpu()], dim=-1)

            # s = copy.deepcopy(_s)
            # output = rearrange(features, "b f t -> b t f").to("cpu")
            #
            # s["sign"] = lzma.compress(pickle.dumps(output))
            # with open(file_name, "wb") as f:
            #     f.write(pickle.dumps(s))
    #  Make file


if __name__ == '__main__':
    #  Get params
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/test/16_eps.slt", help="")
    parser.add_argument("--output_file", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/test/", help="")
    parser.add_argument("--weights", type=str,
                        default="/vol/research/extol/data/Phoenix2014T/Holistic/test/16_eps.slt", help="")
    parser.add_argument("--config", type=str,
                        default='/vol/research/SignRecognition/MS-G3D/config/phoenix_holistic_z.yaml', help="")

    params, _ = parser.parse_known_args()
    print(params.input_file)
    print(params.output_file)
    print(params.weights)
    print(params.config)
    _time = time.time()
    main(params)
    print(f"Done in {time.time() - _time}s")