#!/bin/bash
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh

# exp2022050801
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0508_yolox_s_640_300e_coco.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_s_8x8_300e_coco.py 4

# exp2022050802
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0508_yolox_m_640_300e_coco.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_m_640_300e_coco.py 4


# exp2022051401
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0514_yolox_l_640_r30e_argoverse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_640_r30e_argoverse.py 4

# exp2022051402
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0514_yolox_l_640_41e_argoverse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_640_41e_argoverse.py 4


# exp2022051601
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0516_yolox_l_640_r15e_argoverse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_640_r15e_argoverse.py 4

# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0516_yolox_l_960_r15e_argoverse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_960_r15e_argoverse.py 4

# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0516_yolox_l_1280_r15e_argoverse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_1280_r15e_argoverse.py 4


# exp2022052001
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_640_r15e_argoverse_chip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_640_r15e_argoverse_chip.py 4
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_640_r15e_argoverse_mix.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_640_r15e_argoverse_mix.py 4

# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_960_r15e_argoverse_chip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_960_r15e_argoverse_chip.py 4
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_960_r15e_argoverse_mix.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_960_r15e_argoverse_mix.py 4

# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_1280_r15e_argoverse_chip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_1280_r15e_argoverse_chip.py 4
# nohup bash ./sh_commands/sh_train_yolox_4gpu_01.sh >> ./log_0520_yolox_l_1280_r15e_argoverse_mix.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh configs/yolox/yolox_l_1280_r15e_argoverse_mix.py 4
