# encoding=utf-8
import sys
import utilities as ut

# 脚本作用在于搜索多个影像，并将每个影像都进行分块然后输出，得到训练和验证数据
if __name__ == '__main__':
    raw_dir = sys.argv[1]  # raw影像所在路径
    gt_dir = sys.argv[2]  # gt影像所在路径
    out_dir = sys.argv[3]  # 影像块输出路径

    raw_type = sys.argv[4]  # raw文件格式
    gt_type = sys.argv[5]  # gt文件格式
    out_type = sys.argv[6]  # 输出影像块格式

    block_size = int(sys.argv[7])  # 影像块大小
    num_sample = int(sys.argv[8])  # 影像块个数
    validation_rate = float(sys.argv[9])  # 验证块个数占比

    img_mode = "raw"  # 默认是raw文件
    color_mode = "rgb"  # 默认颜色通道顺序是rgb
    if len(sys.argv) > 10:
        img_mode = sys.argv[10]
        color_mode = sys.argv[11]

    # step1 寻找所有影像
    raw_paths, raw_names, raw_files = ut.findFiles(raw_dir, raw_type)
    gt_paths, gt_names, gt_files = ut.findFiles(gt_dir, gt_type)
    if len(raw_files) != len(gt_files):
        print("Raw file and groundtruth file does not match, please check.")
        exit()

    # step2 遍历影像，每张都进行裁块、增强和输出
    for i in range(len(raw_files)):
        print("processing", i + 1, "/", len(raw_files), "......")

        # step2.1 读取影像与真值
        raw_img, gt_img = ut.readImgPair(raw_files[i], gt_files[i], img_mode)

        # step2.2 随机采样
        raw_blocks, gt_blocks, indices = ut.randomSample(raw_img, gt_img, block_size, num_sample)

        # step2.3 样本增强
        en_raw_blocks, en_gt_blocks, en_indices = ut.enhanceSamples(raw_blocks, gt_blocks, indices)

        # step2.4 输出样本(训练数据+验证数据+索引文件)
        ut.outputSamples(out_dir, out_type, raw_names[i], color_mode, en_raw_blocks, en_gt_blocks, en_indices,
                         validation_rate)

        # step2.5 输出随机采样块的覆盖范围
        ut.outputRangeImg(out_dir, out_type, raw_names[i], raw_img, en_indices, validation_rate)
