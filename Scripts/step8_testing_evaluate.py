# encoding=utf-8
import utilities as ut
import sys
import cv2
import utilities as ut

# 本脚本用于对输出结果进行评估，需要真值影像、噪声影像以及处理后的影像
if __name__ == '__main__':
    gt_path = sys.argv[1]
    input_path = sys.argv[2]
    pred_path = sys.argv[3]

    gt_mode = "raw"  # gt影像是raw还是普通图像，默认为raw
    input_mode = "normal"  # 输入影像默认是普通影像
    pred_mode = "normal"  # 输出影像默认是普通影像
    if len(sys.argv) == 7:
        gt_mode = sys.argv[4]
        input_mode = sys.argv[5]
        pred_mode = sys.argv[6]

    img_gt = ut.readImg(gt_path, gt_mode)
    img_input = ut.readImg(input_path, input_mode)
    img_pred = ut.readImg(pred_path, pred_mode)

    if gt_mode.__contains__("raw") or gt_mode.__contains__("Raw") or gt_mode.__contains__("RAW"):
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)

    if input_mode.__contains__("raw") or input_mode.__contains__("Raw") or input_mode.__contains__("RAW"):
        img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)

    if pred_mode.__contains__("raw") or pred_mode.__contains__("Raw") or pred_mode.__contains__("RAW"):
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)

    mse_g_i = ut.evaluateMSE(img_gt, img_input)
    mse_g_p = ut.evaluateMSE(img_gt, img_pred)

    psnr_g_i = ut.evaluatePSNR(img_gt, img_input)
    psnr_g_p = ut.evaluatePSNR(img_gt, img_pred)

    out_dir = pred_path[:pred_path.rfind("/")]
    out_name = pred_path[pred_path.rfind("/") + 1:].split(".")[0] + "_result.txt"
    out_path = out_dir + "/" + out_name

    print("Metric\tInput\tPred")
    print("MSE" + "\t" + str(mse_g_i) + "\t" + str(mse_g_p))
    print("PSNR" + "\t" + str(psnr_g_i) + "\t" + str(psnr_g_p))

    fout = open(out_path, 'w')
    fout.write("metric\tinput\tpred\n")
    fout.write("mse\t" + str(mse_g_i) + "\t" + str(mse_g_p) + "\n")
    fout.write("psnr\t" + str(psnr_g_i) + "\t" + str(psnr_g_p) + "\n")
    fout.close()
