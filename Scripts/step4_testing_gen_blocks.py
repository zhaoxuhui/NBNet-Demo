# encoding=utf-8
import sys

import utilities as ut

# 本脚本的作用是将输入影像进行分块并输出
if __name__ == '__main__':
    input_dir = sys.argv[1]  # 输入待分割影像目录
    out_dir = sys.argv[2]  # 影像块输出目录
    input_type = sys.argv[3]  # 输入影像文件类型
    output_type = sys.argv[4]  # 输出影像块文件类型

    # 最终影像块宽度=block_width + 2 * overlapping
    # 最终影像块高度=block_height + 2 * overlapping
    block_width = int(sys.argv[5])  # 影像块宽度(不含扩边部分)
    block_height = int(sys.argv[6])  # 影像块高度(不含扩边部分)
    overlapping = int(sys.argv[7])  # 影像块扩边部分大小

    img_mode = "raw"  # 输入影像是raw还是普通图像，默认为raw
    if len(sys.argv) == 9:
        img_mode = sys.argv[8]

    color_mode = "rgb"  # 输入影像颜色通道顺序，默认为rgb
    if len(sys.argv) == 10:
        color_mode = sys.argv[9]

    img_paths, img_names, img_files = ut.findFiles(input_dir, input_type)

    for i in range(len(img_files)):
        print("processing", i + 1, "/", len(img_files))

        tmp_img_path = img_files[i]
        tmp_out_dir = out_dir + "/" + img_names[i].split(".")[0]
        tmp_img_name = img_names[i].split(".")[0]
        ut.isDirExist(tmp_out_dir)

        # 裁剪影像成影像块
        ut.cropImageAndSaveBlocksWithOverlapping(tmp_img_path, block_height, block_width, overlapping,
                                                 tmp_out_dir, tmp_img_name, output_type, img_mode, color_mode)
