import os
import random
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str)
parser.add_argument('-n', default=500, type=int, help='number of val size')
parser.add_argument('--sort', action='store_true')
parser.add_argument('--seed', default=np.random.randint(0, 10000), type=int)
args = parser.parse_args()

pic_ids = list(range(70000))
random.seed(args.seed)
random.shuffle(pic_ids)
num_train = int(len(pic_ids) - args.n)

train_ids = pic_ids[:num_train]
if args.sort:
    train_ids = sorted(train_ids, reverse=False)

train_ids_file = os.path.join(args.o, 'ffhq_train.txt')
with open(train_ids_file, mode='w') as ff:
    for pic_id in train_ids:
        ff.write(f'{pic_id:05d}.png'+'\n')
print(f'Train set: {num_train}/{len(pic_ids)}, will write to {train_ids_file}')


val_ids = pic_ids[num_train:]
if args.sort:
    val_ids = sorted(val_ids, reverse=False)
val_ids_file = os.path.join(args.o, 'ffhq_val.txt')
with open(val_ids_file, mode='w') as ff:
    for pic_id in val_ids:
        ff.write(f'{pic_id:05d}.png'+'\n')
print(f'Val set: {len(pic_ids) - num_train}/{len(pic_ids)}, will write to {val_ids_file}')
