# encoding=utf-8
import sys
import os
import utilities as ut

# 本脚本用于利用网络进行推理
if __name__ == '__main__':
    script_path = sys.argv[1]  # 测试脚本的路径
    img_path = sys.argv[2]  # 输入数据的路径
    model_path = sys.argv[3]  # 模型文件路径
    out_path = sys.argv[4]  # 模型数据输出路径

    ut.isDirExist(out_path)

    command_str = "python3 " + \
                  script_path + \
                  " -d " + img_path + \
                  " -c " + model_path + \
                  " -o " + out_path
    print(command_str)
    print("Start testing......")
    os.system(command_str)
