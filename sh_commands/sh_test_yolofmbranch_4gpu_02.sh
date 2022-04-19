#!/bin/bash
# nohup bash ./sh_commands/sh_test_yolofmbranch_4gpu_02.sh
json_logs_folder="work_dirs_640_archive/faster_rcnn_r50_caffe_fpn_1x_argoverse_hd_640_01100000"
config_file="faster_rcnn_r50_caffe_fpn_1x_argoverse_hd_640.py"
# json_logs_folder="work_dirs_640_archive/fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_640_01080015"
# config_file="fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_640.py"
# json_logs_folder="work_dirs_640_archive/yolof_r50_c5_8x8_1x_argoverse_hd_640_01080011"
# config_file="yolof_r50_c5_8x8_1x_argoverse_hd_640.py"
# json_logs_folder="work_dirs_800_archive/faster_rcnn_r50_caffe_fpn_1x_argoverse_hd_800_01101351"
# config_file="faster_rcnn_r50_caffe_fpn_1x_argoverse_hd_800.py"
# json_logs_folder="work_dirs_800_archive/fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_800_01022353"
# config_file="fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_800.py"
# json_logs_folder="work_dirs_800_archive/fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_800_01080925"
# config_file="fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd_800.py"
# json_logs_folder="work_dirs_800_archive/yolof_r50_c5_8x8_1x_argoverse_hd_800_01022353"
# config_file="yolof_r50_c5_8x8_1x_argoverse_hd_800.py"
# json_logs_folder="work_dirs_800_archive/yolof_r50_c5_8x8_1x_argoverse_hd_800_01080925"
# config_file="yolof_r50_c5_8x8_1x_argoverse_hd_800.py"
# json_logs_folder="work_dirs_800_archive/yolof_r50_c5_8x8_1x_coco_01032354"
# config_file="yolof_r50_c5_8x8_1x_coco.py"
# 
# json_logs_folder="work_dirs_poc_archive/yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc_01051430"
# config_file="yolofmbranchv1v1_r50_c5_8x8_1x_argoverse_hd_poc.py"
checkpoint_file="epoch_12.pth"
gpu_num=4

# single-gpu testing
# CUDA_VISIBLE_DEVICES=7 python tools/test.py \
#     $json_logs_folder/$config_file \
#     $json_logs_folder/$checkpoint_file \
#     --eval bbox

# multi-gpu testing
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
    $json_logs_folder/$config_file \
    $json_logs_folder/$checkpoint_file \
    $gpu_num \
    --eval bbox # --eval-options classwise=True
