#!/bin/bash

total_epochs=100
# edit n_epochs per iter when required
n_epochs=50
# I compute the number of times each has to be called
let "n_calls=$total_epochs/$n_epochs/2"
echo "Number of calls per model alternatively is $n_calls"
let "n_iters=n_calls-1"

# train FetchPush-v1 first and save its buffer. Model is common though from the start, so load all networks
mpirun -np 8 python3 -u train.py --env-name='FetchPush-v1' --save-dir='./outputs/exp_010/' --n-epochs="$n_epochs"\
--save-curve='./plots/exp_010/FetchPush-v1/1/' --cuda 2>&1 | tee logs/exp_010/ani_push_1.log
# train FetchPickAndPlace-v1 next
mpirun -np 8 python3 -u train.py --env-name='FetchPickAndPlace-v1' --save-dir='./outputs/exp_010/' --n-epochs="$n_epochs"\
--pretrain './outputs/exp_010/model.pt' --save-curve='./plots/exp_010/FetchPickAndPlace-v1/1/'\
--cuda 2>&1 | tee logs/exp_010/ani_pick_1.log

# loop through n_iters
for ((i=0;i<$n_iters;i++));
do
  let "num=$i+2"
  echo "Training networks for $num th time"
  # train FetchPush-v1 with previous buffer
  mpirun -np 8 python3 -u train.py --env-name='FetchPush-v1' --save-dir='./outputs/exp_010/' --n-epochs="$n_epochs"\
  --pretrain './outputs/exp_010/model.pt' --load-buffer "./outputs/exp_010/buffer_FetchPush-v1.npy"\
  --save-curve="./plots/exp_010/FetchPush-v1/$num/" --cuda 2>&1 | tee "logs/exp_010/ani_push_$num.log"
  # train FetchPickAndPlace-v1 next with its previous buffer
  mpirun -np 8 python3 -u train.py --env-name='FetchPickAndPlace-v1' --save-dir='./outputs/exp_010/' --n-epochs="$n_epochs"\
  --pretrain './outputs/exp_010/model.pt' --load-buffer "./outputs/exp_010/buffer_FetchPickAndPlace-v1.npy"\
  --save-curve="./plots/exp_010/FetchPickAndPlace-v1/$num/" --cuda 2>&1 | tee "logs/exp_010/ani_pick_$num.log"
done
