# encoding=utf-8
import sys
import cv2
import utilities as ut

# 本脚本用于将分块好的影像块按照SIDD数据集验证数据的格式合并成一个Matlab的mat文件
if __name__ == '__main__':
    input_img_dir = sys.argv[1]  # input影像块所在文件夹
    input_type = sys.argv[2]  # input影像块文件类型
    out_dir = sys.argv[3]  # mat文件输出文件夹

    index_paths, index_names, index_files = ut.findFiles(input_img_dir, ".txt")

    all_blocks = []
    for i in range(len(index_paths)):
        tmp_dir = index_paths[i]
        img_paths, img_names, img_files = ut.findFiles(tmp_dir, input_type)
        for j in range(len(img_files)):
            tmp_block = cv2.imread(img_files[j])
            all_blocks.append(tmp_block)
        print("loaded", i + 1, "/", len(index_paths), ",", len(all_blocks))

    ut.isDirExist(out_dir)
    out_path = out_dir + "/" + "InputBlocksSrgb.mat"
    input_key_name = "InputBlocksSrgb"
    # 将多个影像文件转换成Mat文件 testing-input
    ut.cvtImgs2MatAndSave2(all_blocks, input_key_name, out_path)
