import argparse
from pathlib import Path
import shutil
import sys
import copy


base_configs = {
    "px": Path("/vol/research/SignRecognition/MS-G3D/config/phoenix_holistic_z.yaml"),
    "cl": Path("/vol/research/SignRecognition/MS-G3D/config/chalearn_holistic_z.yaml"),
}
base_submit_path = Path("/vol/research/SignRecognition/MS-G3D/base.submit_file")


def mkdir(path):
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        overwrite_or_quit(path)
        path.mkdir(parents=True)

def overwrite_or_quit(path):
    work_path = path/"work_dir"
    num_epochs = len(list(work_path.glob('*.pkl'))) if work_path.is_dir() else 0
    print(f"Name ({path.stem}) already taken")
    if num_epochs > 0:
        print(f"with {num_epochs} epochs run.\nFor safety You must remove manually")
    elif len(input("Type anything to overwrite, leave blank to quit >")) > 0:
        shutil.rmtree(path)
        print("Directory contents removed")
        return
    sys.exit()


def main(params, seed=111):
    if params.number > 1:
        print(f"Making {params.number} configs")
        commands = []
        for run in range(params.number):
            run_params = copy.deepcopy(params)
            run_params.number = 1
            commands.append(main(run_params, seed=(run + 1) * 111))
        commands = '\n'.join(commands)
        print(f"All commands:{commands}")
        return

    base_config_path = base_configs[params.dataset_type]
    work_path = Path(f"/vol/research/SignRecognition/MS-G3D/work_dirs/")/params.name
    config_path = work_path/"config.yaml"
    submit_path = work_path/"submit.submit_file"
    print(submit_path)
    mkdir(work_path)
    mkdir(work_path/"work_dir")
    shutil.copy(src=base_config_path, dst=config_path)
    shutil.copy(src=base_submit_path, dst=submit_path)
    condor_command = f"condor_submit {submit_path}"
    with open(submit_path, "a") as f:
        f.write(f"""
log    = {work_path/'c$(cluster).p$(process).log'}
output = {work_path/'c$(cluster).p$(process).out'}
error  = {work_path/'c$(cluster).p$(process).error'}
JobBatchName = {params.name}
script = /vol/research/SignRecognition/MS-G3D/main.py --work-dir {work_path/'work_dir'} --config {config_path} --name {params.name} --seed {seed}
queue
# {condor_command}
"""
                )
    print(f"Edit config file then submit with:\n{condor_command}")
    return condor_command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, default="px", help="The name of the experiment")
    parser.add_argument("dataset_type", type=str, default="px", help="Options: px (Phoenix), cl (Chalearn)")
    parser.add_argument("--number", type=int, default=1, help="The number of runs to do")

    params, _ = parser.parse_known_args()
    main(params)


