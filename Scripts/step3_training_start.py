# encoding=utf-8
import sys
import os

# 本脚本用于网络的训练
if __name__ == "__main__":
    script_path = sys.argv[1]  # 训练脚本的路径
    img_path = sys.argv[2]  # 输入数据的路径
    checkpoint_dir = sys.argv[3]  # 训练好的模型输出路径

    print(len(sys.argv))

    # Epoch，默认值为50
    num_epoch = 50
    if len(sys.argv) == 5:
        num_epoch = int(sys.argv[4])

    # Steps per epoch，默认值为200
    num_steps = 200
    if len(sys.argv) == 6:
        num_epoch = int(sys.argv[4])
        num_steps = int(sys.argv[5])

    # Batch size，默认值为32
    batch_size = 32
    if len(sys.argv) == 7:
        num_epoch = int(sys.argv[4])
        num_steps = int(sys.argv[5])
        batch_size = int(sys.argv[6])

    # 模型输出的完整文件路径
    checkpoint_name = "TrainedModel"

    command_str = "python3 " + \
                  script_path + \
                  " -d " + img_path + \
                  " -a " + checkpoint_name + \
                  " --save " + checkpoint_dir + \
                  " --epochs " + str(num_epoch) + \
                  " --steps_per_epoch " + str(num_steps) + \
                  " -b " + str(batch_size) + \
                  " -n 1"
    print(command_str)
    print("Start training......")
    os.system(command_str)
