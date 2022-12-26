#!/bin/bash
# ip102 

port=$(python get_free_port.py)
GPU=2


# fully train ALL 96 CLASSES
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/ip102/ip102_e2e_faster_rcnn_R_50_C4_4x.yaml


#trim model
python tools/trim_detectron_model.py --name ${task}/LR005_CE_BS4_FILOD

task=ip102/90-6
# FIRST STEP
# python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/e2e_faster_rcnn_R_50_C4_4x.yaml
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/${task}/e2e_faster_rcnn_R_50_C4_4x.yaml

# INCREMENTAL STEPS
alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py"
shopt -s expand_aliases

exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n FILOD_noFEAT_UCE_UKD --rpn --uce --dist_type uce --cls 1
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5


task=ip102/50-46
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n FILOD_noFEAT_UCE_UKD --rpn --uce --dist_type uce --cls 1
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1


#--------------------------------------------------------------------
task=10-10
#exp -t ${task} -n ILOD
#exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1


task=15-5
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/${task}/e2e_faster_rcnn_R_50_C4_4x.yaml
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5

task=19-1
#exp -t ${task} -n ILOD
#exp -t ${task} -n FILOD_noFEAT_UCE_UKD --rpn --uce --dist_type uce --cls 1
#exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n ILOD
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n FILOD_noFEAT_UCE_UKD --rpn --uce --dist_type uce --cls 1
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

