#!/bin/bash
# nohup bash ./sh_commands/sh_plot_curve.sh
# json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_1x_argoverse_hd_poc_01082146"
# json_logs="20220108_214652.log.json"
json_logs_folder="work_dirs_poc_archive/yolof_r50_c5_8x8_1x_argoverse_hd_poc_01071635"
json_logs="20220107_163504.log.json"
out="yolof_800_loss.jpg"

python tools/analysis_tools/analyze_logs.py \
    plot_curve $json_logs_folder/$json_logs \
    --keys loss_cls loss_bbox --legend loss_cls loss_bbox \
    --out $json_logs_folder/$out
