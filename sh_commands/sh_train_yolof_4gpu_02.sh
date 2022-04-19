#!/bin/bash
# nohup bash ./sh_commands/sh_train_yolof_4gpu_02.sh

# exp2022010701
# nohup bash ./sh_commands/sh_train_yolof_4gpu_01.sh >> ./log_0107_800_fcos_r50_caffe_fpn.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd.py 4

# exp2022010702
# nohup bash ./sh_commands/sh_train_yolof_4gpu_01.sh >> ./log_0107_640_fcos_r50_caffe_fpn.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_argoverse_hd.py 4

# exp2022010703
# nohup bash ./sh_commands/sh_train_yolof_4gpu_02.sh >> ./log_0107_800_yolof_r50_c5.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train2.sh configs/yolof/yolof_r50_c5_8x8_1x_argoverse_hd.py 4

# exp2022010704
# nohup bash ./sh_commands/sh_train_yolof_4gpu_02.sh >> ./log_0107_640_yolof_r50_c5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train2.sh configs/yolof/yolof_r50_c5_8x8_1x_argoverse_hd.py 4
