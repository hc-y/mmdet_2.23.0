# hc-y_write0112:顺序地先后对 sh_commands/json_logs_folder.txt 中已经训练好的模型进行评估;
import subprocess
import shlex


def main():
    # with open("./sh_commands/json_logs_folder.txt", 'r', encoding="utf8") as f:
    with open("./sh_commands/json_logs_folder.txt", 'r') as f:
        # t = f.readlines()  # each string 后面有 \n
        t = f.read().strip().splitlines()  # each string 后面没有 \n
        # t = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
    checkpoint_file = "epoch_12.pth"
    gpu_num = 8
    for i, json_log_folder in enumerate(t):
        config_py = '_'.join(json_log_folder.split('/')[-1].split('_')[:-1]) + '.py'
        str_call_dist_test_sh = (f'bash tools/dist_test.sh '
            f'{json_log_folder}/{config_py} {json_log_folder}/{checkpoint_file} '
            f'{gpu_num} \"--eval bbox\" \"--eval-options classwise=True\"')  # --eval-options classwise=True
        print('\n' + '---' * 20 + f'{i}-th' + '---' * 20)
        print(f'{json_log_folder}')
        _cur_process = subprocess.run(['pwd'], 
                                stdout=subprocess.PIPE, 
                                universal_newlines=True)
        print(_cur_process.stdout)
        # _cur_process = subprocess.run(shlex.split('bash tools/_01.sh args_01 args_02'), 
        #                 stdout=subprocess.PIPE, 
        #                 universal_newlines=True)
        _cur_process = subprocess.run(shlex.split(str_call_dist_test_sh), 
                                stdout=subprocess.PIPE, 
                                universal_newlines=True)
        print(_cur_process.stdout)


if __name__ == "__main__":
    main()
    print('\nfinish!')
