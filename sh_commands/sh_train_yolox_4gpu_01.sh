#!/bin/bash
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh

# exp2022050801
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0508_yolox_s_640_300e_coco.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_s_8x8_300e_coco.py 4

# exp2022050802
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0508_yolox_m_640_300e_coco.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_m_640_300e_coco.py 4
