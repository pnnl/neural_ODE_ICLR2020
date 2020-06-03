import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('steps', type=int)
parser.add_argument('model')
parser.add_argument('gpu', type=int)
parser.add_argument('-bias', action='store_true')
args = parser.parse_args()
for lr in [0.001, 0.03, 0.01]:
    for k in range(10):
        os.system(f'python system_id.py --gpu {args.gpu} -lr {lr} -nsteps {args.steps} '
                  f'-cell_type {args.model} {["", "-bias"][args.bias]} -exp sysid_{args.model}_{args.steps}_{args.bias} -run {k}_{lr}_{args.model} -location nonlin_simple/mlruns')
