#!/bin/bash
# nohup bash ./sh_commands/sh_test_yolox_4gpu_01.sh
json_logs_folder="work_dirs/yolox_s_8x8_rf300e_argoverse_05042135"
# config_file="yolox_s_8x8_rf300e_argoverse.py"
# config_file="configs/yolox/yolox_s_8x8_rf300e_argoverse.py"
config_file="configs/yolox/yoloxv1_s_8x8_300e_argoverse.py"
checkpoint_file="epoch_420.pth"
gpu_num=4

# single-gpu testing
# CUDA_VISIBLE_DEVICES=7 python tools/test.py \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     --eval bbox

# single-gpu testing
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    $config_file \
    $json_logs_folder/$checkpoint_file \
    --eval bbox #--show-dir ""$json_logs_folder"_results"

# multi-gpu testing
# bash tools/dist_test.sh \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     $gpu_num \
#     --eval bbox --eval-options classwise=True
