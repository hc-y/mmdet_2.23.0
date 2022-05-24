#!/bin/bash
# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh

# exp2022051501
# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0515_fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640.py 4

# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0515_fcos_r50_caffe_fpn_gn-head_r1x_argoverse_960.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_960.py 4

# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0515_fcos_r50_caffe_fpn_gn-head_r1x_argoverse_1280.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_1280.py 4

# exp2022051502
# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0515_fcos_r101_caffe_fpn_gn-head_r1x_argoverse_640.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r101_caffe_fpn_gn-head_r1x_argoverse_640.py 4


# exp2022052001
# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0520_fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640_chip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640_chip.py 4
# nohup bash ./sh_commands/sh_train_fcos_4gpu_01.sh >> ./log_0520_fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640_mix.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640_mix.py 4
