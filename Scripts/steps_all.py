# encoding=utf-8
import os

# 本脚本用于批量进行网络训练和推理
if __name__ == '__main__':
    command_training = "python3 steps_training.py"
    command_testing = "python3 steps_testing.py"

    os.system(command_training)
    os.system(command_testing)
