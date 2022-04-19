#!/bin/bash
# hc-y_write0111:there are some bugs in this shell script;
# nohup bash ./sh_commands/sh_test_yolofmbranch_8gpu_01loop.sh
checkpoint_file="epoch_12.pth"
gpu_num=8
printf "*************************************\n"
echo "for line in cat json_logs_folder.txt"
SAVEIFS=$IFS
IFS=$(echo -en "\n")
count=0
var=1
pwd
for line in $(cat ./sh_commands/json_logs_folder.txt)
    do
        count=$(( $count + $var ))
        echo $line;
        echo $count
        # echo $(`expr $count % 2`)
        if [ `expr $count % 2` == 1 ]
        then
            json_logs_folder=$line
            # echo $json_logs_folder
        elif [ `expr $count % 2` == 0 ]
        then
            # config_file=$line
            # echo $json_logs_folder
            # echo $json_logs_folder/$config_file
            echo "\n"
            # bash tools/dist_test.sh \
            #     $json_logs_folder/$config_file \
            #     $json_logs_folder/$checkpoint_file \
            #     $gpu_num \
            #     --eval bbox # --eval-options classwise=True
        else
            echo "ed"
        fi
    done
IFS=$SAVEIFS 
