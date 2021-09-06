# encoding=utf-8
import os
import cv2
import utilities as ut

# 本脚本用于批量进行推理
if __name__ == '__main__':
    fs = cv2.FileStorage("testing.yml", cv2.FILE_STORAGE_READ)

    test_script_path = fs.getNode("test_script_path").string()
    pretrained_model_path = fs.getNode("pretrained_model_path").string()
    pred_out_path = fs.getNode("pred_out_path").string()

    input_img_dir = fs.getNode("input_img_dir").string()
    input_img_type = fs.getNode("input_img_type").string()

    out_block_dir = fs.getNode("out_block_dir").string()
    out_block_type = fs.getNode("out_block_type").string()

    block_width = int(fs.getNode("block_width").real())
    block_height = int(fs.getNode("block_height").real())
    overlapping = int(fs.getNode("overlapping").real())

    img_mode = fs.getNode("img_mode").string()
    color_mode = fs.getNode("color_mode").string()

    # step1
    command1 = "python3 step4_testing_gen_blocks.py " + \
               input_img_dir + " " + \
               out_block_dir + " " + \
               input_img_type + " " + \
               out_block_type + " " + \
               str(block_width) + " " + \
               str(block_height) + " " + \
               str(overlapping)
    print(command1)
    os.system(command1)

    # step2
    command2 = "python3 step5_testing_gen_mat.py " + \
               out_block_dir + " " + \
               out_block_type + " " + \
               out_block_dir
    print(command2)
    os.system(command2)

    # step3
    ut.isDirExist(pred_out_path)
    command3 = "python3 step6_testing_start.py " + \
               test_script_path + " " + \
               out_block_dir + " " + \
               pretrained_model_path + " " + \
               pred_out_path
    print(command3)
    os.system(command3)

    # step4
    model_out_path1 = pred_out_path + "/input"
    model_out_path2 = pred_out_path + "/pred"

    merge_out_path1 = pred_out_path + "/input_merge"
    merge_out_path2 = pred_out_path + "/pred_merge"

    ut.isDirExist(merge_out_path1)
    ut.isDirExist(merge_out_path2)

    print("Merging input noisy blocks ......")
    command4_1 = "python3 step7_testing_merge_blocks.py " + \
                 out_block_dir + " " + \
                 model_out_path1 + " " + \
                 merge_out_path1 + " " + \
                 out_block_type
    print(command4_1)
    os.system(command4_1)

    print("Merging output pred blocks ......")
    command4_2 = "python3 step7_testing_merge_blocks.py " + \
                 out_block_dir + " " + \
                 model_out_path2 + " " + \
                 merge_out_path2 + " " + \
                 out_block_type
    print(command4_2)
    os.system(command4_2)
