#!/usr/bin/env python3
from dataset import SIDDValData, SIDDValData2, SIDDValData3
from model import UNetD
import megengine.data as data
from utils import batch_PSNR
from tqdm import tqdm
import argparse
import pickle
import megengine
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os


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

def test(args):
    # modifiy for our data(testing)
    valid_dataset = SIDDValData3(args.data)
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=1, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        num_workers=8,
    )
    model = UNetD(3)
    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    model.load_state_dict(state["state_dict"])
    model.eval()

    def valid_step(image, label):
        pred = model(image)
        pred = image - pred

        # added for saving results
        # ---------------------------------------------------------------
        # 对于非0-1值截断
        image_numpy = image.detach().numpy().squeeze().swapaxes(0,2).clip(0,1)
        pred_numpy = pred.detach().numpy().squeeze().swapaxes(0,2).clip(0,1)

        image_numpy_rotate = cv2.rotate(image_numpy, cv2.ROTATE_90_CLOCKWISE)
        pred_numpy_rotate = cv2.rotate(pred_numpy, cv2.ROTATE_90_CLOCKWISE)
        image_numpy_rotate_flip = cv2.flip(image_numpy_rotate, 1)
        pred_numpy_rotate_flip = cv2.flip(pred_numpy_rotate, 1)

        image_numpy = (image_numpy_rotate_flip * 255 ).astype('uint8')
        pred_numpy = (pred_numpy_rotate_flip * 255 ).astype('uint8')

        tmp_t = str(time.time())

        raw_dir = args.output + "/input"
        pred_dir = args.output + "/pred"

        isDirExist(raw_dir)
        isDirExist(pred_dir)

        cv2.imwrite(raw_dir + "/"+tmp_t+"_raw.png",image_numpy)
        cv2.imwrite(pred_dir+"/"+tmp_t+"_pred.png",pred_numpy)
        # ---------------------------------------------------------------

        psnr_it = batch_PSNR(pred, label)
        return psnr_it

    def valid(func, data_queue):
        psnr_v = 0.
        for step, (image, label) in tqdm(enumerate(data_queue)):
            image = megengine.tensor(image)
            label = megengine.tensor(label)
            psnr_it = func(image, label)
            psnr_v += psnr_it
        psnr_v /= step + 1
        return psnr_v

    psnr_v = valid(valid_step, valid_dataloader)
    print("PSNR: {:.3f}".format(psnr_v.item()) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegEngine NBNet")
    parser.add_argument("-d", "--data", default="/data/sidd", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument("-c", "--checkpoint", help="path to checkpoint")
    parser.add_argument("-o", "--output", help="output path")
    args = parser.parse_args()
    test(args)



# vim: ts=4 sw=4 sts=4 expandtab
