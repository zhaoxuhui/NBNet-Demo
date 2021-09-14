# encoding=utf-8
from matplotlib import pyplot as plt  # 可视化相关
import rawpy  # Raw图解析相关
import numpy as np  # 矩阵运算相关
import random  # 随机数生成相关
import cv2  # 影像读写相关
import os  # 系统操作相关
from scipy.io import loadmat  # mat文件读取
from scipy.io import savemat  # mat文件保存
import time  # 时间相关


def readRaw(img_path):
    # 读取Raw图
    img = rawpy.imread(img_path)

    # 直接解析
    out_pp = img.postprocess(use_camera_wb=True)

    return out_pp


def readImg(img_path, img_mode):
    print(img_mode)
    # 如果是Raw的文件，用Rawpy读取，否则的话用OpenCV读取
    if img_mode.__contains__("raw") or img_mode.__contains__("RAW") or img_mode.__contains__("Raw"):
        img = readRaw(img_path)
    else:
        img = cv2.imread(img_path)
    return img


def readImgPair(raw_path, gt_path, img_mode):
    raw_img = readImg(raw_path, img_mode)
    gt_img = readImg(gt_path, img_mode)
    return raw_img, gt_img


def visualizeRaw(img, title="Raw image"):
    # 可视化
    plt.figure(1)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def randomSample(img1, img2, block_size, num_sample):
    blocks1 = []
    blocks2 = []
    indices = []
    counter = 0

    img_width = img1.shape[1]
    img_height = img1.shape[0]

    while counter < num_sample:
        tmp_x_start = random.randint(0, img_width - block_size)
        tmp_y_start = random.randint(0, img_height - block_size)
        tmp_x_end = tmp_x_start + block_size
        tmp_y_end = tmp_y_start + block_size

        block1 = img1[tmp_y_start:tmp_y_end, tmp_x_start:tmp_x_end]
        block2 = img2[tmp_y_start:tmp_y_end, tmp_x_start:tmp_x_end]

        blocks1.append(block1)
        blocks2.append(block2)
        indices.append([tmp_x_start, tmp_x_end, tmp_y_start, tmp_y_end, 0])
        counter += 1
        if counter % 50 == 0:
            print("Added", counter, "/", num_sample)
    return blocks1, blocks2, indices


def enhanceSamples(raw_blocks, gt_blocks, indices):
    enhanced_raw_blocks = []
    enhanced_gt_blocks = []
    enhanced_indices = []
    for i in range(len(raw_blocks)):
        tmp_raw_block = raw_blocks[i]
        tmp_gt_block = gt_blocks[i]

        tmp_raw_block_1 = cv2.flip(tmp_raw_block, 1)  # 水平翻转
        tmp_gt_block_1 = cv2.flip(tmp_gt_block, 1)

        tmp_raw_block_2 = cv2.flip(tmp_raw_block, 0)  # 竖直翻转
        tmp_gt_block_2 = cv2.flip(tmp_gt_block, 0)

        tmp_raw_block_3 = cv2.flip(tmp_raw_block_2, 0)  # 竖直+水平翻转
        tmp_gt_block_3 = cv2.flip(tmp_gt_block_2, 0)

        enhanced_raw_blocks.append(tmp_raw_block)
        enhanced_raw_blocks.append(tmp_raw_block_1)
        enhanced_raw_blocks.append(tmp_raw_block_2)
        enhanced_raw_blocks.append(tmp_raw_block_3)

        enhanced_gt_blocks.append(tmp_gt_block)
        enhanced_gt_blocks.append(tmp_gt_block_1)
        enhanced_gt_blocks.append(tmp_gt_block_2)
        enhanced_gt_blocks.append(tmp_gt_block_3)

        enhanced_indices.append(indices[i])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 1])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 2])
        enhanced_indices.append([indices[i][0], indices[i][1], indices[i][2], indices[i][3], 3])
        if (i + 1) % 50 == 0:
            print("Enhanced", i + 1, "/", len(raw_blocks))
    return enhanced_raw_blocks, enhanced_gt_blocks, enhanced_indices


def outputSamples(out_dir, out_type, raw_name, color_mode, raw_blocks, gt_blocks, indices, vali_rate):
    train_input_dir = out_dir + "/train/input"
    train_groundtruth_dir = out_dir + "/train/groundtruth"
    isDirExist(train_input_dir)
    isDirExist(train_groundtruth_dir)

    vali_input_dir = out_dir + "/validation/input"
    vali_groundtruth_dir = out_dir + "/validation/groundtruth"
    isDirExist(vali_input_dir)
    isDirExist(vali_groundtruth_dir)

    index_dir = out_dir + "/indices"
    isDirExist(index_dir)

    if out_type[0] != ".":
        out_type = "." + out_type

    vali_num = int(vali_rate * len(raw_blocks)) + 1
    vali_step = int(len(raw_blocks) / vali_num)

    fout = open(index_dir + "/" + raw_name.split(".")[0] + "_indices.txt", "w")
    fout.write("# number\tx_start\tx_end\ty_start\ty_end\ttype\tpattern\n")
    for i in range(len(raw_blocks)):
        if i % vali_step == 0:
            tmp_gt_path = vali_groundtruth_dir
            tmp_in_path = vali_input_dir
            tmp_flag = 'v'
        else:
            tmp_gt_path = train_groundtruth_dir
            tmp_in_path = train_input_dir
            tmp_flag = 't'

        # 如果彩色通道顺序是RGB，就转换一下，否则不用额外操作
        if color_mode.__contains__("rgb") or color_mode.__contains__("RGB") or color_mode.__contains__("Rgb"):
            tmp_raw = cv2.cvtColor(raw_blocks[i], cv2.COLOR_RGB2BGR)
            tmp_gt = cv2.cvtColor(gt_blocks[i], cv2.COLOR_RGB2BGR)
        else:
            tmp_raw = raw_blocks[i]
            tmp_gt = gt_blocks[i]

        cv2.imwrite(tmp_in_path + "/" + raw_name.split(".")[0] + "_" + i.__str__().zfill(5) + "_input" + out_type,
                    tmp_raw)
        cv2.imwrite(tmp_gt_path + "/" + raw_name.split(".")[0] + "_" + i.__str__().zfill(5) + "_gt" + out_type, tmp_gt)
        fout.write(i.__str__().zfill(5) + "\t" +
                   str(indices[i][0]) + "\t" + str(indices[i][1]) + "\t" +
                   str(indices[i][2]) + "\t" + str(indices[i][3]) + "\t" +
                   tmp_flag + "\t" +
                   str(indices[i][4]) + "\n")
        if (i + 1) % 50 == 0:
            print("Saved", raw_name, ":", i + 1, "/", len(raw_blocks))
    fout.close()


def drawRange(raw_img, indices, vali_rate):
    img = np.zeros([raw_img.shape[0], raw_img.shape[1]], np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    vali_num = int(vali_rate * len(indices)) + 1
    vali_step = int(len(indices) / vali_num)

    for i in range(len(indices)):
        if i % vali_step == 0:
            color = (100, 150, 200)
        else:
            color = (128, 200, 180)
        start_x = indices[i][0]
        end_x = indices[i][1]
        start_y = indices[i][2]
        end_y = indices[i][3]
        cv2.rectangle(img_rgb, (start_x, start_y), (end_x, end_y), color, thickness=2)
    return img_rgb


def outputRangeImg(out_dir, out_type, raw_name, raw_img, indices, validation_rate):
    range_out_dir = out_dir + "/range"
    isDirExist(range_out_dir)

    if out_type != ".":
        out_type = "." + out_type

    img_range = drawRange(raw_img, indices, validation_rate)
    cv2.imwrite(range_out_dir + "/" + raw_name.split(".")[0] + "_block_range" + out_type, img_range)


def calcMSE(gt_img, input_img):
    in_band = input_img.astype(np.float)
    gt_band = gt_img.astype(np.float)

    diff = np.power(in_band - gt_band, 2)
    mse = np.sum(diff) / (input_img.shape[0] * input_img.shape[1])
    return mse


def evaluateMSE(gt_img, input_img):
    # for rgb color images
    if len(input_img.shape) == 3:
        in_band_r = input_img[:, :, 0]
        in_band_g = input_img[:, :, 1]
        in_band_b = input_img[:, :, 2]
        gt_band_r = gt_img[:, :, 0]
        gt_band_g = gt_img[:, :, 1]
        gt_band_b = gt_img[:, :, 2]

        mse_r = calcMSE(gt_band_r, in_band_r)
        mse_g = calcMSE(gt_band_g, in_band_g)
        mse_b = calcMSE(gt_band_b, in_band_b)
        mse_mean = (mse_r + mse_g + mse_b) / 3
        return mse_mean
    # for grayscale(single band) images
    else:
        mse = calcMSE(gt_img, input_img)
        return mse


def evaluatePSNR(gt_img, input_img, bit_level=8):
    mean_mse = evaluateMSE(gt_img, input_img)
    if mean_mse == 0:
        mean_mse = 0.000000001
    psnr = 10 * np.log10((np.power(2, 8) - 1) ** 2 / mean_mse)
    return psnr


def findFiles(root_dir, filter_type, reverse=False):
    """
    在指定目录查找指定类型文件

    :param root_dir: 查找目录
    :param filter_type: 文件类型
    :param reverse: 是否返回倒序文件列表，默认为False
    :return: 路径、名称、文件全路径

    """

    print("Finding files ends with \'" + filter_type + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter_type):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print(names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    if reverse:
        paths.reverse()
        names.reverse()
        files.reverse()
    return paths, names, files


def isDirExist(path='output'):
    """
    判断指定目录是否存在，如果存在返回True，否则返回False并新建目录

    :param path: 指定目录
    :return: 判断结果

    """

    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def cvtImgs2Mat(img_dir, file_type, img_key_name):
    paths, names, files = findFiles(img_dir, file_type)

    imgs = []
    for i in range(len(files)):
        tmp_img = cv2.imread(files[i])
        imgs.append(tmp_img)

    img_width = imgs[0].shape[1]
    img_height = imgs[0].shape[0]
    num_channel = imgs[0].shape[2]
    num_imgs = len(imgs)
    print("Target shape:[", num_imgs, img_height, img_width, num_channel, "]")
    img_mat = np.zeros([num_imgs, img_height, img_width, num_channel], np.uint8)

    for i in range(len(imgs)):
        img_mat[i, :, :, :] = imgs[i]

    # print(checkItem(imgs, img_mat, 300))
    # checkAndVisualizeItem(imgs, img_mat, 300)

    img_dict = {img_key_name: img_mat,
                '__header__': 'Matlab MAT-file, Created by Xuhui Zhao on ' + time.ctime(),
                '__version__': '1.0',
                '__globals__': ''}
    return img_dict


def cvtImgs2MatAndSave(img_dir, file_type, img_key_name, out_path):
    paths, names, files = findFiles(img_dir, file_type)

    imgs = []
    for i in range(len(files)):
        tmp_img = cv2.imread(files[i])
        imgs.append(tmp_img)

    img_width = imgs[0].shape[1]
    img_height = imgs[0].shape[0]
    num_channel = imgs[0].shape[2]
    num_imgs = len(imgs)
    print("Target shape:[", num_imgs, img_height, img_width, num_channel, "]")
    img_mat = np.zeros([num_imgs, img_height, img_width, num_channel], np.uint8)

    for i in range(len(imgs)):
        img_mat[i, :, :, :] = imgs[i]

    # print(checkItem(imgs, img_mat, 300))
    # checkAndVisualizeItem(imgs, img_mat, 300)

    img_dict = {img_key_name: img_mat,
                '__header__': 'Matlab MAT-file, Created by Xuhui Zhao on ' + time.ctime(),
                '__version__': '1.0',
                '__globals__': ''}

    savemat(out_path, img_dict)
    return img_dict


def cvtImgs2MatAndSave2(imgs, img_key_name, out_path):
    img_width = imgs[0].shape[1]
    img_height = imgs[0].shape[0]
    num_channel = imgs[0].shape[2]
    num_imgs = len(imgs)
    print("Target shape:[", num_imgs, img_height, img_width, num_channel, "]")
    img_mat = np.zeros([num_imgs, img_height, img_width, num_channel], np.uint8)

    for i in range(len(imgs)):
        img_mat[i, :, :, :] = imgs[i]

    img_dict = {img_key_name: img_mat,
                '__header__': 'Matlab MAT-file, Created by Xuhui Zhao on ' + time.ctime(),
                '__version__': '1.0',
                '__globals__': ''}

    savemat(out_path, img_dict)
    return img_dict


def cvtMat2Imgs(mat_path, img_key_name):
    imgs = []
    img_mat = loadmat(mat_path)
    img_data = img_mat[img_key_name]
    for i in range(img_data.shape[0]):
        imgs.append(img_data[i, :, :, :])
    return imgs


def cvtMat2ImgsAndSave(mat_path, img_key_name, img_out_dir, img_type):
    imgs = []
    img_mat = loadmat(mat_path)
    print(img_mat.keys())
    img_data = img_mat[img_key_name]
    for i in range(img_data.shape[0]):
        imgs.append(img_data[i, :, :, :])

    for i in range(len(imgs)):
        cv2.imwrite(img_out_dir + "/" + str(i).zfill(5) + "." + img_type, imgs[i])
    return imgs


def cropImage(img, block_height, block_width):
    # 先获得影像的长宽
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 再计算按照给定的影像块大小完全覆盖全图需要多少行列
    if img_height % block_height != 0:
        row_num = int(img_height / block_height) + 1
    else:
        row_num = int(img_height / block_height)
    if img_width % block_width != 0:
        col_num = int(img_width / block_width) + 1
    else:
        col_num = int(img_width / block_width)

    # 再计算根据指定的影像块大小和行列数，全图影像应该有多大以及与原图的差异
    target_height = row_num * block_height
    target_width = col_num * block_width
    diff_height = target_height - img_height
    diff_width = target_width - img_width

    # 根据尺寸差异计算上下左右应该各自向外扩展多少
    if diff_height % 2 != 0:
        padding_top = int(diff_height / 2) + 1
    else:
        padding_top = int(diff_height / 2)
    padding_bottom = diff_height - padding_top
    if diff_width % 2 != 0:
        padding_left = int(diff_width / 2) + 1
    else:
        padding_left = int(diff_width / 2)
    padding_right = diff_width - padding_left

    # 对图像进行扩边
    img_padding = cv2.copyMakeBorder(img,
                                     padding_top, padding_bottom,
                                     padding_left, padding_right,
                                     cv2.BORDER_REFLECT)

    # 依次对每个影像块进行处理
    img_blocks = []
    block_indices = []
    counter = 0
    for i in range(row_num):
        for j in range(col_num):
            tmp_start_y = i * block_height
            tmp_start_x = j * block_width
            tmp_end_y = tmp_start_y + block_height
            tmp_end_x = tmp_start_x + block_width
            tmp_block = img_padding[tmp_start_y:tmp_end_y, tmp_start_x:tmp_end_x, :]
            img_blocks.append(tmp_block)
            block_indices.append([counter, i, j, tmp_start_x, tmp_start_y, tmp_end_x, tmp_end_y])
            counter += 1

    block_param = [img_height, img_width,
                   target_height, target_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right]

    return img_blocks, block_indices, block_param


def cropImageWithOverlapping(img, block_height, block_width, overlapping):
    # 先获得影像的长宽
    img_height = img.shape[0]
    img_width = img.shape[1]

    # 再计算按照给定的影像块大小完全覆盖全图需要多少行列
    if img_height % block_height != 0:
        row_num = int(img_height / block_height) + 1
    else:
        row_num = int(img_height / block_height)
    if img_width % block_width != 0:
        col_num = int(img_width / block_width) + 1
    else:
        col_num = int(img_width / block_width)

    # 再计算根据指定的影像块大小和行列数，全图影像应该有多大以及与原图的差异
    target_height = row_num * block_height
    target_width = col_num * block_width
    diff_height = target_height - img_height
    diff_width = target_width - img_width

    # 根据尺寸差异计算上下左右应该各自向外扩展多少
    if diff_height % 2 != 0:
        padding_top = int(diff_height / 2) + 1
    else:
        padding_top = int(diff_height / 2)
    padding_bottom = diff_height - padding_top
    if diff_width % 2 != 0:
        padding_left = int(diff_width / 2) + 1
    else:
        padding_left = int(diff_width / 2)
    padding_right = diff_width - padding_left

    # 对图像进行扩边
    img_padding = cv2.copyMakeBorder(img,
                                     padding_top, padding_bottom,
                                     padding_left, padding_right,
                                     cv2.BORDER_REFLECT)

    # 在之前的基础上进一步向外扩，满足overlapping的要求
    padding_overlap_top = padding_top + overlapping
    padding_overlap_bottom = padding_bottom + overlapping
    padding_overlap_left = padding_left + overlapping
    padding_overlap_right = padding_right + overlapping

    img_padding_overlap = cv2.copyMakeBorder(img,
                                             padding_overlap_top, padding_overlap_bottom,
                                             padding_overlap_left, padding_overlap_right,
                                             cv2.BORDER_REFLECT)
    img_padding_overlap_height = img_padding_overlap.shape[0]
    img_padding_overlap_width = img_padding_overlap.shape[1]

    # 依次对每个影像块进行处理
    img_blocks = []
    block_indices = []
    counter = 0
    for i in range(row_num):
        for j in range(col_num):
            # 这四个坐标是在处理完padding的影像上的坐标(对应img_padding)
            tmp_start_y_padding = i * block_height
            tmp_start_x_padding = j * block_width
            tmp_end_y_padding = tmp_start_y_padding + block_height
            tmp_end_x_padding = tmp_start_x_padding + block_width

            # 这四个坐标是每个影像块扩展以后在再一次扩边影像上的坐标(对应img_padding_overlap)
            tmp_start_x_overlap = tmp_start_x_padding
            tmp_start_y_overlap = tmp_start_y_padding
            tmp_end_x_overlap = tmp_end_x_padding + 2 * overlapping
            tmp_end_y_overlap = tmp_end_y_padding + 2 * overlapping

            # 不包含扩展区域的影像块
            tmp_block_padding = img_padding[tmp_start_y_padding:tmp_end_y_padding,
                                tmp_start_x_padding:tmp_end_x_padding, :]
            # 包括扩展区域的影像块
            tmp_block_overlap = img_padding_overlap[tmp_start_y_overlap:tmp_end_y_overlap,
                                tmp_start_x_overlap:tmp_end_x_overlap, :]

            img_blocks.append(tmp_block_overlap)
            block_indices.append([counter,
                                  i, j,
                                  tmp_start_x_overlap, tmp_start_y_overlap,
                                  tmp_end_x_overlap, tmp_end_y_overlap,
                                  tmp_start_x_padding, tmp_start_y_padding,
                                  tmp_end_x_padding, tmp_end_y_padding])
            counter += 1

    block_param = [img_height, img_width,
                   target_height, target_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    return img_blocks, block_indices, block_param


def saveBlocks(out_dir, out_filename, out_type, img_blocks, block_indices, block_param):
    if not out_type.__contains__("."):
        out_type = "." + out_type

    fout = open(out_dir + "/indices.txt", "w")
    fout.write("original image height:" + str(block_param[0]) + "\n")
    fout.write("original image width:" + str(block_param[1]) + "\n")
    fout.write("padding image height:" + str(block_param[2]) + "\n")
    fout.write("padding image width:" + str(block_param[3]) + "\n")
    fout.write("row num:" + str(block_param[4]) + "\n")
    fout.write("col num:" + str(block_param[5]) + "\n")
    fout.write("block height:" + str(block_param[6]) + "\n")
    fout.write("block width:" + str(block_param[7]) + "\n")
    fout.write("padding top:" + str(block_param[8]) + "\n")
    fout.write("padding bottom:" + str(block_param[9]) + "\n")
    fout.write("padding left:" + str(block_param[10]) + "\n")
    fout.write("padding right:" + str(block_param[11]) + "\n")
    fout.write("file name format:filename_rowindex_colindex.filetype\n")
    fout.write("Block indices in padding image:\n")
    fout.write("number\trow index\tcol index\tx_start\ty_start\tx_end\ty_end\n")

    for i in range(len(img_blocks)):
        index_num = block_indices[i][0]
        block_row = block_indices[i][1]
        block_col = block_indices[i][2]
        start_x = block_indices[i][3]
        start_y = block_indices[i][4]
        end_x = block_indices[i][5]
        end_y = block_indices[i][6]
        block_img = img_blocks[i]

        fout.write(index_num + "\t" +
                   str(block_row) + "\t" + str(block_col) + "\t" +
                   str(start_x) + "\t" + str(start_y) + "\t" +
                   str(end_x) + "\t" + str(end_y) + "\n")
        cv2.imwrite(out_dir + "/" + out_filename + "_" +
                    str(block_row).zfill(4) + "_" + str(block_col).zfill(4) + out_type,
                    block_img)
    fout.close()


def saveBlocksWithOverlapping(out_dir, out_filename, out_type, img_blocks, block_indices, block_param, color_mode):
    if not out_type.__contains__("."):
        out_type = "." + out_type

    fout = open(out_dir + "/indices.txt", "w")
    fout.write("original image height:" + str(block_param[0]) + "\n")
    fout.write("original image width:" + str(block_param[1]) + "\n")
    fout.write("padding image height:" + str(block_param[2]) + "\n")
    fout.write("padding image width:" + str(block_param[3]) + "\n")
    fout.write("row num:" + str(block_param[4]) + "\n")
    fout.write("col num:" + str(block_param[5]) + "\n")
    fout.write("block height:" + str(block_param[6]) + "\n")
    fout.write("block width:" + str(block_param[7]) + "\n")
    fout.write("padding top:" + str(block_param[8]) + "\n")
    fout.write("padding bottom:" + str(block_param[9]) + "\n")
    fout.write("padding left:" + str(block_param[10]) + "\n")
    fout.write("padding right:" + str(block_param[11]) + "\n")
    fout.write("padding overlap top:" + str(block_param[12]) + "\n")
    fout.write("padding overlap bottom:" + str(block_param[13]) + "\n")
    fout.write("padding overlap left:" + str(block_param[14]) + "\n")
    fout.write("padding overlap right:" + str(block_param[15]) + "\n")
    fout.write("image overlap height:" + str(block_param[16]) + "\n")
    fout.write("image overlap width:" + str(block_param[17]) + "\n")
    fout.write("overlapping:" + str(block_param[18]) + "\n")

    fout.write("file name format:filename_rowindex_colindex.filetype\n")
    fout.write("Block indices in padding image:\n")

    fout.write("number\trow index\tcol index\t"
               "x_start(overlap)\ty_start(overlap)\tx_end(overlap)\ty_end(overlap)\t"
               "x_start(padding)\ty_start(padding)\tx_end(padding)\ty_end(padding)\n")

    for i in range(len(img_blocks)):
        index_num = block_indices[i][0]
        block_row = block_indices[i][1]
        block_col = block_indices[i][2]
        start_x_overlap = block_indices[i][3]
        start_y_overlap = block_indices[i][4]
        end_x_overlap = block_indices[i][5]
        end_y_overlap = block_indices[i][6]
        start_x_padding = block_indices[i][7]
        start_y_padding = block_indices[i][8]
        end_x_padding = block_indices[i][9]
        end_y_padding = block_indices[i][10]
        block_img = img_blocks[i]

        fout.write(str(index_num) + "\t" +
                   str(block_row) + "\t" + str(block_col) + "\t" +
                   str(start_x_overlap) + "\t" + str(start_y_overlap) + "\t" +
                   str(end_x_overlap) + "\t" + str(end_y_overlap) + "\t" +
                   str(start_x_padding) + "\t" + str(start_y_padding) + "\t" +
                   str(end_x_padding) + "\t" + str(end_y_padding) + "\n")

        # 如果彩色通道顺序是RGB，就转换一下，否则不用额外操作
        if color_mode.__contains__("rgb") or color_mode.__contains__("RGB") or color_mode.__contains__("Rgb"):
            block_img = cv2.cvtColor(block_img, cv2.COLOR_RGB2BGR)
        else:
            block_img = block_img

        cv2.imwrite(out_dir + "/" + out_filename + "_" +
                    str(block_row).zfill(4) + "_" + str(block_col).zfill(4) + out_type,
                    block_img)

    fout.close()


def cropImageAndSaveBlocks(img_path, block_height, block_width, out_dir, out_filename, out_type):
    img = cv2.imread(img_path)
    img_blocks, block_indices, block_param = cropImage(img, block_height, block_width)
    saveBlocks(out_dir, out_filename, out_type, img_blocks, block_indices, block_param)


def cropImageAndSaveBlocksWithOverlapping(img_path, block_height, block_width, overlapping,
                                          out_dir, out_filename, out_type, img_mode, color_mode):
    img = readImg(img_path, img_mode)
    img_blocks, block_indices, block_param = cropImageWithOverlapping(img, block_height, block_width, overlapping)
    saveBlocksWithOverlapping(out_dir, out_filename, out_type, img_blocks, block_indices, block_param, color_mode)


def loadIndexFile(file_path):
    fin = open(file_path, "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])
    padding_overlap_top = int(fin.readline().strip().split(":")[1])
    padding_overlap_bottom = int(fin.readline().strip().split(":")[1])
    padding_overlap_left = int(fin.readline().strip().split(":")[1])
    padding_overlap_right = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_height = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_width = int(fin.readline().strip().split(":")[1])
    overlapping = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start_overlap = int(line_parts[3])
        y_start_overlap = int(line_parts[4])
        x_end_overlap = int(line_parts[5])
        y_end_overlap = int(line_parts[6])
        x_start_padding = int(line_parts[7])
        y_start_padding = int(line_parts[8])
        x_end_padding = int(line_parts[9])
        y_end_padding = int(line_parts[10])
        block_indices.append([index_num,
                              row_index, col_index,
                              x_start_overlap, y_start_overlap,
                              x_end_overlap, y_end_overlap,
                              x_start_padding, y_start_padding,
                              x_end_padding, y_end_padding])

        line = fin.readline().strip()
    fin.close()
    return block_param, block_indices


def loadBlocks(block_dir, block_type):
    fin = open(block_dir + "/indices.txt", "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start = int(line_parts[3])
        y_start = int(line_parts[4])
        x_end = int(line_parts[5])
        y_end = int(line_parts[6])
        block_indices.append([index_num, row_index, col_index, x_start, y_start, x_end, y_end])

        line = fin.readline().strip()

    img_blocks = []
    paths, names, files = findFiles(block_dir, block_type)
    for i in range(len(files)):
        img_block = cv2.imread(files[i])
        img_blocks.append(img_block)

    fin.close()

    return img_blocks, block_indices, block_param


def loadBlocksWithOverlapping(block_dir, block_type):
    fin = open(block_dir + "/indices.txt", "r")
    original_img_height = int(fin.readline().strip().split(":")[1])
    original_img_width = int(fin.readline().strip().split(":")[1])
    padding_img_height = int(fin.readline().strip().split(":")[1])
    padding_img_width = int(fin.readline().strip().split(":")[1])
    row_num = int(fin.readline().strip().split(":")[1])
    col_num = int(fin.readline().strip().split(":")[1])
    block_height = int(fin.readline().strip().split(":")[1])
    block_width = int(fin.readline().strip().split(":")[1])
    padding_top = int(fin.readline().strip().split(":")[1])
    padding_bottom = int(fin.readline().strip().split(":")[1])
    padding_left = int(fin.readline().strip().split(":")[1])
    padding_right = int(fin.readline().strip().split(":")[1])
    padding_overlap_top = int(fin.readline().strip().split(":")[1])
    padding_overlap_bottom = int(fin.readline().strip().split(":")[1])
    padding_overlap_left = int(fin.readline().strip().split(":")[1])
    padding_overlap_right = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_height = int(fin.readline().strip().split(":")[1])
    img_padding_overlap_width = int(fin.readline().strip().split(":")[1])
    overlapping = int(fin.readline().strip().split(":")[1])

    block_param = [original_img_height, original_img_width,
                   padding_img_height, padding_img_width,
                   row_num, col_num,
                   block_height, block_width,
                   padding_top, padding_bottom,
                   padding_left, padding_right,
                   padding_overlap_top, padding_overlap_bottom,
                   padding_overlap_left, padding_overlap_right,
                   img_padding_overlap_height, img_padding_overlap_width,
                   overlapping]

    fin.readline()
    fin.readline()
    fin.readline()

    block_indices = []
    line = fin.readline().strip()
    while line:
        line_parts = line.split("\t")
        index_num = int(line_parts[0])
        row_index = int(line_parts[1])
        col_index = int(line_parts[2])
        x_start_overlap = int(line_parts[3])
        y_start_overlap = int(line_parts[4])
        x_end_overlap = int(line_parts[5])
        y_end_overlap = int(line_parts[6])
        x_start_padding = int(line_parts[7])
        y_start_padding = int(line_parts[8])
        x_end_padding = int(line_parts[9])
        y_end_padding = int(line_parts[10])
        block_indices.append([index_num,
                              row_index, col_index,
                              x_start_overlap, y_start_overlap,
                              x_end_overlap, y_end_overlap,
                              x_start_padding, y_start_padding,
                              x_end_padding, y_end_padding])

        line = fin.readline().strip()

    img_blocks = []
    paths, names, files = findFiles(block_dir, block_type)
    for i in range(len(files)):
        img_block = cv2.imread(files[i])
        img_blocks.append(img_block)

    fin.close()

    return img_blocks, block_indices, block_param


def mergeBlocks(img_blocks, block_indices, block_param):
    padding_img_height = block_param[2]
    padding_img_width = block_param[3]
    padding_img = np.zeros([padding_img_height, padding_img_width, 3], np.uint8)

    for i in range(len(img_blocks)):
        tmp_start_x = block_indices[i][2]
        tmp_start_y = block_indices[i][3]
        tmp_end_x = block_indices[i][4]
        tmp_end_y = block_indices[i][5]
        padding_img[tmp_start_y:tmp_end_y, tmp_start_x:tmp_end_x, :] = img_blocks[i]
    padding_top = block_param[8]
    padding_bottom = block_param[9]
    padding_left = block_param[10]
    padding_right = block_param[11]

    crop_img = padding_img[padding_top:padding_img_height - padding_bottom,
               padding_left:padding_img_width - padding_right, :]
    return padding_img, crop_img


def mergeBlocksWithOverlapping(img_blocks, block_indices, block_param):
    overlapping_img_height = block_param[16]
    overlapping_img_width = block_param[17]
    overlapping_img = np.zeros([overlapping_img_height, overlapping_img_width, 3], np.uint8)

    padding_img_height = block_param[2]
    padding_img_width = block_param[3]
    padding_img = np.zeros([padding_img_height, padding_img_width, 3], np.uint8)

    padding_top = block_param[8]
    padding_bottom = block_param[9]
    padding_left = block_param[10]
    padding_right = block_param[11]

    overlapping = block_param[18]

    for i in range(len(img_blocks)):
        start_x_overlap = block_indices[i][3]
        start_y_overlap = block_indices[i][4]
        end_x_overlap = block_indices[i][5]
        end_y_overlap = block_indices[i][6]

        start_x_padding = block_indices[i][7]
        start_y_padding = block_indices[i][8]
        end_x_padding = block_indices[i][9]
        end_y_padding = block_indices[i][10]

        block_overlap = img_blocks[i]
        block_padding = img_blocks[i][overlapping:block_overlap.shape[0] - overlapping,
                        overlapping:block_overlap.shape[1] - overlapping, :]

        # 对于重叠影像，直接贴过来（会有拼接缝，如果有更好的融合方法可以尝试）
        overlapping_img[start_y_overlap:end_y_overlap, start_x_overlap:end_x_overlap, :] = block_overlap
        # 对于非重叠影像，裁剪影像块之后再贴过来（无拼接缝）
        padding_img[start_y_padding:end_y_padding, start_x_padding:end_x_padding, :] = block_padding

    # 对于原始影像，直接在padding影像上裁剪即可
    original_img = padding_img[padding_top:padding_img.shape[0] - padding_bottom,
                   padding_left:padding_img.shape[1] - padding_right, :]
    return overlapping_img, padding_img, original_img


def mergeBlocksAndSaveImage(block_dir, block_type, out_dir, out_type):
    if not out_type.__contains__("."):
        out_type = "." + out_type
    img_blocks, block_indices, block_param = loadBlocks(block_dir, block_type)
    padding_img, crop_img = mergeBlocks(img_blocks, block_indices, block_param)
    cv2.imwrite(out_dir + "/" + "merge_padding" + out_type, padding_img)
    cv2.imwrite(out_dir + "/" + "merge_original" + out_type, crop_img)


def mergeBlocksAndSaveImageWithOverlapping(block_dir, block_type, out_dir, out_type):
    if not out_type.__contains__("."):
        out_type = "." + out_type
    img_blocks, block_indices, block_param = loadBlocksWithOverlapping(block_dir, block_type)
    overlapping_img, crop_img, original_img = mergeBlocksWithOverlapping(img_blocks, block_indices, block_param)
    cv2.imwrite(out_dir + "/" + "merge_overlapping" + out_type, overlapping_img)
    cv2.imwrite(out_dir + "/" + "merge_cropping" + out_type, crop_img)
    cv2.imwrite(out_dir + "/" + "merge_original" + out_type, original_img)
