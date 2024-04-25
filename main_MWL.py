from argparse import ArgumentParser
from utils import get_task_chunk, log2csv, ensure_path
import numpy as np
from Task import LOSO
import os

parser = ArgumentParser()
parser.add_argument('--task-ID', type=int, default=0, help='The experiment index. By setting this,'
                                                           ' you can run LOSO on several machines by '
                                                           'assign n subjects as one group to be the test subjects')
parser.add_argument('--task-step', type=int, default=5, help='Every n subjects is within one task')
parser.add_argument('--full-run', type=int, default=0, help='If it is set as 1, you will run LOSO on the same machine.')
######## Data ########
parser.add_argument('--dataset', type=str, default='MWL')
parser.add_argument('--subjects', type=int, default=36)
parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
parser.add_argument('--label-type', type=str, default='MW')
parser.add_argument('--num-chan', type=int, default=19)  # 16 for TSception, 19 for others
parser.add_argument('--num-time', type=int, default=2000)
parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
parser.add_argument('--overlap', type=float, default=0)
parser.add_argument('--sampling-rate', type=int, default=500)
parser.add_argument('--data-format', type=str, default='eeg')
######## Training Process ########
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--additional-epoch', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--val-rate', type=float, default=0.2)

parser.add_argument('--save-path', default='./save/')
parser.add_argument('--load-path', default='./data_processed/') # change this
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mixed-precision', type=int, default=0)
######## Model Parameters ########
parser.add_argument('--model', type=str, default='Deformer')
parser.add_argument('--graph-type', type=str, default='BL', choices=['LGG-G', 'LGG-F', 'LGG-H', 'TS', 'BL'])
parser.add_argument('--kernel-length', type=int, default=51)
parser.add_argument('--T', type=int, default=64)
parser.add_argument('--AT', type=int, default=16)
parser.add_argument('--num-layers', type=int, default=6)

args = parser.parse_args()
if args.full_run:
    TASK_CHUNK = np.arange(args.subjects)
else:
    TASK_CHUNK = get_task_chunk(np.arange(args.subjects), step=args.task_step)
logs_name = 'logs_{}_{}'.format(args.dataset, args.model)

for sub in TASK_CHUNK[args.task_ID]:
    results = LOSO(
        test_idx=[sub], subjects=list(range(args.subjects)),
        experiment_ID='sub{}'.format(sub), args=args, logs_name=logs_name
    )
    log_path = os.path.join(os.getcwd(), logs_name, 'sub{}'.format(sub))
    ensure_path(log_path)
    log2csv(os.path.join(log_path, 'result.csv'), results[0])
