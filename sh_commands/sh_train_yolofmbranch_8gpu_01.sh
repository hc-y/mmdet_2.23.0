#!/bin/bash
# nohup bash ./sh_commands/sh_train_yolofmbranch_8gpu_01.sh >> ./log_0109_800_MBranchVxvx_poc.log 2>&1 &
# echo "current time is :"
# date;
# sleep 1.5h;
# pwd
# exp2022010901
# bash ./tools/dist_train.sh configs/yolof/yolofmbranchv1v2_r50_c5_8x8_1x_argoverse_hd_poc.py 8

# echo "current time is :"
# date;
# sleep 0.1h;
# pwd
# exp2022010902
# bash ./tools/dist_train.sh configs/yolof/yolofmbranchv2v1_r50_c5_8x8_1x_argoverse_hd_poc.py 8

# echo "current time is :"
# date;
# sleep 0.1h;
# pwd
# exp2022010903
# bash ./tools/dist_train.sh configs/yolof/yolofmbranchv2v2_r50_c5_8x8_1x_argoverse_hd_poc.py 8

echo "current time is :"
date;
# sleep 0.1h;
pwd
# exp2022011101
bash ./tools/dist_train.sh configs/yolof/yolofmbranchv2v2_r50_c5_8x8_2x_argoverse_hd_poc_grad.py 8

echo "current time is :"
date;
sleep 0.1h;
pwd
# exp2022011102
bash ./tools/dist_train.sh configs/yolof/yolofv0v1_r50_c5_8x8_2x_argoverse_hd_poc.py 8
