# encoding=utf-8
import os
import cv2

# 本脚本用于批量训练网络
if __name__ == '__main__':
    fs = cv2.FileStorage("training.yml", cv2.FILE_STORAGE_READ)

    input_img_dir = fs.getNode("input_img_dir").string()
    input_img_type = fs.getNode("input_img_type").string()

    gt_img_dir = fs.getNode("gt_img_dir").string()
    gt_img_type = fs.getNode("gt_img_type").string()

    block_img_dir = fs.getNode("block_img_dir").string()
    block_img_type = fs.getNode("block_img_type").string()

    train_script_path = fs.getNode("train_script_path").string()

    block_size = int(fs.getNode("block_size").real())
    block_num = int(fs.getNode("block_num").real())
    vali_rate = fs.getNode("vali_rate").real()

    img_mode = fs.getNode("img_mode").string()
    color_mode = fs.getNode("color_mode").string()

    epoch_num = int(fs.getNode("epoch_num").real())
    step_num = int(fs.getNode("step_num").real())
    batch_size = int(fs.getNode("batch_size").real())

    # step1
    command1 = "python3 step1_training_gen_blocks.py " + \
               input_img_dir + " " + \
               gt_img_dir + " " + \
               block_img_dir + " " + \
               input_img_type + " " + \
               gt_img_type + " " + \
               block_img_type + " " + \
               str(block_size) + " " + \
               str(block_num) + " " + \
               str(vali_rate)
    print(command1)
    os.system(command1)

    # step2
    command2 = "python3 step2_training_gen_vali_mat.py " + \
               block_img_dir + "/validation/input" + " " + \
               block_img_dir + "/validation/groundtruth" + " " + \
               block_img_type + " " + \
               block_img_type + " " + \
               block_img_dir + "/validation"
    print(command2)
    os.system(command2)

    # step3
    command3 = "python3 step3_training_start.py " + \
               train_script_path + " " + \
               block_img_dir + " " + \
               block_img_dir[:block_img_dir.rfind("/")] + " " + \
               str(epoch_num) + " " + \
               str(step_num) + " " + \
               str(batch_size)

    print(command3)
    os.system(command3)
