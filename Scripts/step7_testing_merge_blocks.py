# encoding=utf-8
import utilities as ut
import sys
import cv2

# 本脚本用于将网络输出的影像块重新合并成一个大图
if __name__ == '__main__':
    block_dir = sys.argv[1]  # 原始影像块文件夹
    model_out_dir = sys.argv[2]  # 模型输出影像块文件夹
    merge_out_dir = sys.argv[3]  # 合并影像输出文件夹
    merge_out_type = sys.argv[4]  # 输出合并影像的文件类型

    if not merge_out_type.__contains__("."):
        merge_out_type = "." + merge_out_type

    index_paths, index_names, index_files = ut.findFiles(block_dir, ".txt")
    block_paths, block_names, block_files = ut.findFiles(model_out_dir, ".png")

    img_blocks = []
    for i in range(len(block_files)):
        img_block = cv2.imread(block_files[i])
        img_blocks.append(img_block)

    for i in range(len(index_files)):
        print("processing image:", i + 1, "/", len(index_files))

        tmp_index_file = index_files[i]
        block_param, block_indices = ut.loadIndexFile(tmp_index_file)
        block_num = len(block_indices)
        start_index = i * block_num
        end_index = (i + 1) * block_num

        overlapping_img, crop_img, original_img = ut.mergeBlocksWithOverlapping(img_blocks[start_index:end_index],
                                                                                block_indices, block_param)

        img_name = index_paths[i].split("/")[-2]
        tmp_out_dir = merge_out_dir + "/" + img_name
        ut.isDirExist(tmp_out_dir)
        out_file_path1 = tmp_out_dir + "/merge_" + img_name + "_overlap" + merge_out_type
        out_file_path2 = tmp_out_dir + "/merge_" + img_name + "_crop" + merge_out_type
        out_file_path3 = tmp_out_dir + "/merge_" + img_name + "_original" + merge_out_type

        cv2.imwrite(out_file_path1, overlapping_img)
        cv2.imwrite(out_file_path2, crop_img)
        cv2.imwrite(out_file_path3, original_img)
