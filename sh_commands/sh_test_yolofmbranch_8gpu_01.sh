#!/bin/bash
# nohup bash ./sh_commands/sh_test_yolofmbranch_8gpu_01.sh
# json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_1x_coco_01032354"
# config_file=".py"
# json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_1x_argoverse_hd_poc_01071635"
# config_file="yolof_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_1x_argoverse_hd_poc_01082146"
# config_file="yolof_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_2x_argoverse_hd_poc_01091740"
# config_file="yolof_r50_c5_8x8_2x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofv0v1_r50_c5_8x8_1x_argoverse_hd_poc_01092053"
# config_file="yolofv0v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc_01051430"
# config_file="yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc_01071654"
# config_file="yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc_01082005"
# config_file="yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc_01072145"
# config_file="yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc_01090001"
# config_file="yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc_grad_01090456"
# config_file="yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc_grad.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc_01090943"
# config_file="yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc_grad_01091215"
# config_file="yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc_grad.py"
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc_01091059"
# config_file="yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc.py"
json_logs_folder="work_dirs_poc_archive/yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc_grad_01091334"
config_file="yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc_grad.py"
checkpoint_file="epoch_12.pth"
gpu_num=8

# single-gpu testing
# CUDA_VISIBLE_DEVICES=7 python tools/test.py \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     --eval bbox

# single-gpu testing
CUDA_VISIBLE_DEVICES=7 python tools/test.py \
    $json_logs_folder/$config_file \
    $json_logs_folder/$checkpoint_file \
    --eval bbox --show-dir ""$json_logs_folder"_results"

# multi-gpu testing
# bash tools/dist_test.sh \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     $gpu_num \
#     --eval bbox --eval-options classwise=True
