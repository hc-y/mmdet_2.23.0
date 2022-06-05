#!/bin/bash
# nohup bash ./sh_commands/sh_test_fcos_4gpu_02.sh
json_logs_folder="work_dirs/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_960_mix_05291317"
checkpoint_file="epoch_9_.pth"
# json_logs_folder="work_dirs/fcos_r50_caffe_fpn_gn-head_r1x_argoverse_640_05190938"
config_file="configs/fcos/fcosv1_r50_caffe_fpn_gn-head_r1x_argoverse_640.py"
# checkpoint_file="epoch_12_.pth"
gpu_num=4

# single-gpu testing
# CUDA_VISIBLE_DEVICES=7 python tools/test.py \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     --eval bbox

# single-gpu testing
CUDA_VISIBLE_DEVICES=3 python tools/test.py \
    $config_file \
    $json_logs_folder/$checkpoint_file \
    --eval bbox --eval-options classwise=True #--show-dir ""$json_logs_folder"_results"

# multi-gpu testing
# bash tools/dist_test.sh \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     $gpu_num \
#     --eval bbox --eval-options classwise=True
