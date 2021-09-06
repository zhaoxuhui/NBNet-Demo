# encoding=utf-8
import sys
import utilities as ut

# 本脚本用于实现将分块好的影像块按照SIDD数据集验证数据的格式合并成一个Matlab的mat文件
if __name__ == '__main__':
    input_img_dir = sys.argv[1]  # input影像块所在文件夹
    gt_img_dir = sys.argv[2]  # gt影像块所在文件夹
    input_type = sys.argv[3]  # input影像块文件类型
    gt_type = sys.argv[4]  # gt影像块文件类型
    out_dir = sys.argv[5]  # mat文件输出文件夹

    # 文件默认名如下(与SIDD一致)
    out_input_mat_name = out_dir + "/ValidationNoisyBlocksSrgb.mat"
    out_gt_mat_name = out_dir + "/ValidationGtBlocksSrgb.mat"
    # 如果设置了文件名，则用设置的名字
    if len(sys.argv) == 8:
        out_input_mat_name = out_dir + "/" + sys.argv[6]  # input mat输出名称
        out_gt_mat_name = out_dir + "/" + sys.argv[7]  # gt mat输出名称
    # 如果设置的文件名没有后缀名，再加上
    if not (out_input_mat_name.__contains__(".mat") or out_input_mat_name.__contains__(".MAT")):
        out_input_mat_name = out_input_mat_name + ".mat"
    if not (out_gt_mat_name.__contains__(".mat") or out_gt_mat_name.__contains__(".MAT")):
        out_gt_mat_name = out_gt_mat_name + ".mat"

    # mat文件的key name(与SIDD一致)
    input_key_name = "ValidationNoisyBlocksSrgb"
    gt_key_name = "ValidationGtBlocksSrgb"
    if len(sys.argv) == 10:
        input_key_name = sys.argv[8]  # input mat的key name
        gt_key_name = sys.argv[9]  # gt mat的key name

    # 将多个影像文件转换成Mat文件 validation-input
    ut.cvtImgs2MatAndSave(input_img_dir, input_type, input_key_name, out_input_mat_name)

    # 将多个影像文件转换成Mat文件 validation-groundtruth
    ut.cvtImgs2MatAndSave(gt_img_dir, gt_type, gt_key_name, out_gt_mat_name)
