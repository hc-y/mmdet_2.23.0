#!/bin/bash
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_02.sh

# exp2022010801
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_01.sh >> ./log_0108_800_MBranchV1v2_FusV1v1_poc.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./../tools/dist_train.sh configs/yolof/yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc.py 4
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_02.sh >> ./log_0108_800_MBranchV1v2_FusV1v1_poc.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./../tools/dist_train2.sh configs/yolof/yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc.py 4

# exp2022010802
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_01.sh >> ./log_0108_800_MBranchV2v1_FusV2v1_poc.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./../tools/dist_train.sh configs/yolof/yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc.py 4
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_02.sh >> ./log_0108_800_MBranchV2v1_FusV2v1_poc.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./../tools/dist_train2.sh configs/yolof/yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc.py 4

# exp2022010803
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_01.sh >> ./log_0108_800_MBranchV2v2_FusV3v1_poc.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./../tools/dist_train.sh configs/yolof/yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc.py 4
# nohup bash ./sh_commands/sh_train_yolofmbranch_4gpu_02.sh >> ./log_0108_800_MBranchV2v2_FusV3v1_poc.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./../tools/dist_train2.sh configs/yolof/yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc.py 4
